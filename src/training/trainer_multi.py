"""
统一训练器 —— 同一份内存数据喂 1..N 个模型，每个模型可单阶段或多阶段。

一个入口覆盖三种用法
--------------------
  1. **单独训练一个模型**（单阶段）
        runs 只放一项即可，等同于传统单模型训练脚本。

  2. **一起训练多个模型**（共享数据管道）
        runs 放 N 项。特征提取每个 batch 只做一次，N 个模型依次前向/反向；
       N 个模型看到的增强样本**完全一致**，对照实验更严格。

  3. **多阶段训练**（如双路融合的三阶段：冻结静态 → 解冻微调）
       给某个 run 写 stages 列表即可。阶段切换时自动重建该 run 的
       优化器与调度器，并切换静态分支的冻结状态。

       注意：`mode='static'` 的模型永远不会被冻结（冻结会让它没有可训练参数）。

为什么共享管道最省时间
----------------------
本项目的瓶颈是 **CPU 数据管道**（detect_corners / pseudo_segment /
signatory 五次签名 / 63 次 equalizeHist / 两个几何变换），不是 GPU。
复刻仓库 README 也明确写着「GPU 长期处于饥饿状态」。
所以开 N 个进程只会让它们抢同一批 CPU 核，总时间并不会减少。

    一个 WPTTDataset（内存里只有一份）
              ↓
    一个 DataLoader（特征提取每个 batch 只做一次）
              ↓
       ┌──────┴──────┐
    batch  →  模型A  → loss → backward → step
       └──────→  模型B  → loss → backward → step

显存峰值 ≈ max(各模型激活) 而非求和：每个模型 backward 完立刻释放自己的计算图。

与 train_dual.py 的关系
-----------------------
`train_dual.py` 是历史入口，只做「单模型 + 三阶段」。
本文件完全覆盖它的能力（给单个 run 写 stages 即可），并额外支持多模型共享。
新项目建议直接用本文件；`train_dual.py` 保留是为了 `run_ablation.sh` 不失效。
"""

from __future__ import annotations

import os
import sys
import csv
import time
import gc
import math
import random as _random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import WPTTDataset, STATIC_FEATURE_VERSION
from src.models.dual_stream import build_model
from src.evaluation.evaluator import (
    evaluate_page_level, evaluate_page_level_multi,
    evaluate_page_level_20test_multi,
)
from src.training.trainer_dual import (
    _unpack_batch, _worker_init_fn, _pad_to_96, _ensure_dir, _loader_context,
)
from src.data.dataloader_utils import make_loader as _make_loader

_np = np


# ----------------------------------------------------------------------
# 单个「运行」的状态
# ----------------------------------------------------------------------

class RunSpec:
    """一个待训练模型的配置与运行状态。"""

    def __init__(self, cfg: dict):
        self.raw = dict(cfg)
        self.name = cfg['name']
        self.mode = cfg.get('mode', 'static')
        self.arch = cfg.get('arch', 'convnext')
        self.prefix = cfg.get('prefix', f"{self.arch}_{self.mode}")
        self.lr = float(cfg.get('lr', 5e-4))
        self.optimizer_name = str(cfg.get('optimizer', 'adamw')).lower()
        self.weight_decay = float(cfg.get('weight_decay', 0.05))
        self.warmup_epochs = int(cfg.get('warmup_epochs', 1))
        self.static_init_path = cfg.get('static_init_path')
        self.eval_frequency = int(cfg.get('eval_frequency', 1))
        self.extra = cfg.get('model_overrides', {}) or {}

        # 运行时状态
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.stages = []
        self.schedule = {}          # global_epoch -> (stage_idx, epoch_in_stage)
        self.total_epochs = 0
        self.cur_stage = None
        self.best_acc = 0.0
        self.best_epoch = 0

    @property
    def needs_dynamic(self) -> bool:
        return self.mode in ('dynamic', 'dual')

    def best_path(self, ckpt_dir: str) -> str:
        return os.path.join(ckpt_dir, f'{self.prefix}_best.pth')

    def __repr__(self):
        return (f"<{self.name}: arch={self.arch} mode={self.mode} "
                f"lr={self.lr:.2e} opt={self.optimizer_name}>")


def resolve_run_stages(run: RunSpec, config: dict):
    """
    解析某个 run 的训练阶段。

    优先级：run 自己的 stages → config['stages_by_mode'][mode] → 单阶段默认。
    返回项保证含 name / epochs / lr / warmup_epochs / freeze_static /
    static_lr_scale。
    """
    stages = run.raw.get('stages')
    if stages is None:
        by_mode = config.get('stages_by_mode') or {}
        stages = by_mode.get(run.mode)

    if stages is None:
        # 单阶段：直接用 run 自己的 lr / warmup
        return [{
            'name': f'{run.name}_single',
            'epochs': int(config.get('epochs', 50)),
            'lr': run.lr,
            'warmup_epochs': run.warmup_epochs,
            'freeze_static': False,
            'static_lr_scale': 1.0,
        }]

    out = []
    for i, st in enumerate(stages):
        st = dict(st)
        st.setdefault('name', f'{run.name}_stage{i+1}')
        if st.get('lr') is None:
            st['lr'] = run.lr
        st['lr'] = float(st['lr'])
        st.setdefault('warmup_epochs', run.warmup_epochs)
        st.setdefault('freeze_static', False)
        st.setdefault('static_lr_scale', 1.0)
        if int(st.get('epochs', 0)) <= 0:
            raise ValueError(f"[{run.name}] 阶段 {st['name']} 的 epochs 必须 > 0")
        out.append(st)
    return out


def build_run_schedule(run: RunSpec):
    """预先算出 global_epoch -> (stage_idx, epoch_in_stage)，便于断点续训。"""
    schedule = {}
    g = 0
    for si, st in enumerate(run.stages):
        for ei in range(1, int(st['epochs']) + 1):
            g += 1
            schedule[g] = (si, ei)
    run.schedule = schedule
    run.total_epochs = g


# ----------------------------------------------------------------------
# 优化器 / 调度器
# ----------------------------------------------------------------------

def build_optimizer(run: RunSpec, stage: dict):
    """
    按阶段构造优化器。

    阶段里给了 freeze_static 时切换静态分支的冻结状态；解冻后静态分支
    用 static_lr_scale 倍的学习率，避免把已收敛的权重整体带偏。
    """
    model = run.model
    model.set_static_frozen(bool(stage.get('freeze_static', False)))

    lr = float(stage['lr'])
    scale = float(stage.get('static_lr_scale', 1.0))

    static_ids = set()
    if getattr(model, 'static_branch', None) is not None:
        static_ids = {id(p) for p in model.static_branch.parameters()}

    static_params, other_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (static_params if id(p) in static_ids else other_params).append(p)

    groups = []
    if static_params:
        groups.append({'params': static_params, 'lr': lr * scale, 'name': 'static'})
    if other_params:
        groups.append({'params': other_params, 'lr': lr, 'name': 'other'})
    if not groups:
        raise RuntimeError(f"[{run.name}] 没有可训练参数，检查 freeze_static 设置")

    if run.optimizer_name == 'adam':
        # 原论文 / 复刻仓库的配方：Adam，无 weight decay
        return optim.Adam(groups, lr=lr)
    if run.optimizer_name == 'adamw':
        return optim.AdamW(groups, lr=lr, weight_decay=run.weight_decay,
                           betas=(0.9, 0.999))
    raise ValueError(f"[{run.name}] 未知优化器: {run.optimizer_name}")


def build_scheduler(run: RunSpec, stage: dict, steps_per_epoch: int):
    """阶段内的 warmup + 余弦退火（iteration 级 step）。"""
    optimizer = run.optimizer
    epochs = int(stage['epochs'])
    warmup_epochs = int(stage.get('warmup_epochs', 0))
    base_lr = float(stage['lr'])
    eta_min = max(base_lr * 0.01, 1e-6)

    warmup = max(int(warmup_epochs * steps_per_epoch), 1)
    cosine = max(int((epochs - warmup_epochs) * steps_per_epoch), 1)

    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(cosine, 1), eta_min=eta_min)

    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-6, end_factor=1.0,
                     total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=cosine, eta_min=eta_min),
        ],
        milestones=[warmup],
    )


# ----------------------------------------------------------------------
# 日志：训练过程的所有指标都落进 CSV
# ----------------------------------------------------------------------

#: 每个模型在逐轮 CSV 里的列（顺序即写入顺序）
PER_RUN_COLUMNS = [
    'stage',            # 阶段号.阶段内轮次/阶段总轮数，如 3.7/25
    'train_loss',       # 本轮训练集平均 loss
    'train_char_acc',   # 本轮训练集伪字符级 top-1
    'train_char_top5',  # 本轮训练集伪字符级 top-5
    'test_char_acc',    # 测试集伪字符级 top-1（apply_drop=False，确定性）
    'test_char_top5',   # 测试集伪字符级 top-5
    'page_acc',         # 页面级投票 top-1  ← 论文主指标
    'page_acc_top5',    # 页面级投票 top-5
    'best_page_acc',    # 截至本轮的历史最佳
    'is_best',          # 本轮是否刷新最佳（1/0）
    'lr',               # 主参数组学习率
    'lr_static',        # 静态分支学习率（阶段3 解冻后才有意义）
    'grad_norm',        # 梯度范数（裁剪前），监视训练稳定性
    'gpu_mem_peak_mb',  # 本轮 GPU 显存峰值
]

GLOBAL_COLUMNS = ['epoch', 'timestamp', 'epoch_time_sec',
                  'compute_time_sec', 'data_time_sec']


def _log_header(runs):
    header = list(GLOBAL_COLUMNS)
    for r in runs:
        header += [f'{r.name}_{c}' for c in PER_RUN_COLUMNS]
    return header


def _init_log(path, runs, reset=False):
    """
    准备逐轮日志。

    Args:
        reset: True 时**清空重建**（全新训练必须这样，否则本轮数据会
            追加到上一次实验的旧行后面，两个实验混在同一个 CSV 里，
            列名相同但语义不同，事后无法区分）。
            False 时保留已有内容（断点续训要接着写）。
    """
    _ensure_dir(os.path.dirname(path), "训练日志")
    exists = os.path.exists(path)

    if exists and not reset:
        return 'append'

    if exists and reset:
        # 旧日志改名留档，而不是直接删掉 —— 万一里面是有用的历史数据
        stamp = time.strftime('%Y%m%d_%H%M%S')
        backup = f'{path}.bak_{stamp}'
        os.replace(path, backup)
        print(f"🗂  旧日志已留档: {os.path.basename(backup)}")

    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(_log_header(runs))
    print(f"📝 已创建训练日志: {path}")
    print(f"   每个模型 {len(PER_RUN_COLUMNS)} 列: {', '.join(PER_RUN_COLUMNS)}")
    return 'reset' if exists else 'new'


def _append_log(path, row):
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(row)


def _dump_config(config, runs, ckpt_dir, group):
    """
    把**解析后的完整配置 + 运行环境**落盘。

    这一步对复现性很关键：CSV 里只有指标，没有「这些指标是在什么配置下
    跑出来的」。把配置与版本信息一起存档，以后回看不会抓瞎。
    """
    import json
    import platform

    payload = {
        'group': group,
        'runs': [
            {
                'name': r.name, 'arch': r.arch, 'mode': r.mode,
                'prefix': r.prefix, 'lr': r.lr,
                'optimizer': r.optimizer_name,
                'weight_decay': r.weight_decay,
                'warmup_epochs': r.warmup_epochs,
                'static_init_path': r.static_init_path,
                'eval_frequency': r.eval_frequency,
                'stages': r.stages,
                'total_epochs': r.total_epochs,
                'model_overrides': r.extra,
            }
            for r in runs
        ],
        'config': {k: v for k, v in config.items() if k != 'runs'},
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu': (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            'cudnn': torch.backends.cudnn.version(),
        },
    }
    path = os.path.join(ckpt_dir, f'{group}_config.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"📝 配置与环境已存档: {path}")
    return path


def _dump_summary(path, runs, ckpt_dir, total_wall_sec, max_epochs):
    """每个模型一行的小结 CSV：论文里做表最方便。"""
    _ensure_dir(os.path.dirname(path), "小结 CSV")
    exists = os.path.exists(path)
    with open(path, 'a' if exists else 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['run_name', 'arch', 'mode', 'prefix',
                        'params_total_M', 'params_trainable_M',
                        'best_page_acc', 'best_epoch', 'epochs_configured',
                        'wall_time_sec', 'best_ckpt', 'finished_at'])
        for r in runs:
            w.writerow([
                r.name, r.arch, r.mode, r.prefix,
                f"{r.model.num_total_params()/1e6:.4f}",
                f"{r.model.num_trainable_params()/1e6:.4f}",
                f"{r.best_acc:.6f}", r.best_epoch,
                max_epochs, f"{total_wall_sec:.1f}",
                os.path.basename(r.best_path(ckpt_dir)),
                time.strftime('%Y-%m-%d %H:%M:%S'),
            ])
    return path


# ----------------------------------------------------------------------
# 主训练
# ----------------------------------------------------------------------

def train_multi(config: dict):
    """
    Args:
        config: 见 configs/multi_arch.yaml。
            必需 'runs'（列表）。1 项 = 单独训练，N 项 = 共享管道一起训练。

    Returns:
        {'runs': {name: {...}}, 'combined_path':..., 'log_path':..., 'epochs': n}
    """
    run_cfgs = config.get('runs') or []
    if not run_cfgs:
        raise ValueError("config['runs'] 为空，没有要训练的模型")
    train_start = time.time()
    runs = [RunSpec(c) for c in run_cfgs]
    if len({r.name for r in runs}) != len(runs):
        raise ValueError("runs 里有重名，name 必须唯一")
    if len({r.prefix for r in runs}) != len(runs):
        raise ValueError("runs 里有重复的 checkpoint 前缀，会互相覆盖")

    BATCH_SIZE = config.get('batch_size', 100)
    NUM_WORKERS = config.get('num_workers', 16)
    # 早停已移除，原因见文件末尾「关于早停」的说明。
    # 旧的 config 里如果还留着 patience 键，这里读出来只用于打印提示。
    _LEGACY_PATIENCE = config.get('patience', None)
    if _LEGACY_PATIENCE is not None:
        print(f"ℹ️  配置里的 patience={_LEGACY_PATIENCE} 已忽略 —— 早停已移除，"
              f"训练会跑满 epochs（原因见 README「关于早停」）")
    GRAD_CLIP = config.get('grad_clip', 5.0)
    SEQ_LEN = config.get('seq_len', 200)
    DYNAMIC_SPEED_CLIP = config.get('dynamic_speed_clip', 10.0)
    # 几何增强（平移+旋转+缩放）作用在**坐标**上，论文要求在算签名之前做。
    # 评估时必须关闭，否则测试集不再确定（每轮看到的增强不同）。
    AUGMENT_POINTS = bool(config.get('augment_points', True))
    # 位图是否作为第 0 通道。论文两处表述矛盾（详见 dataset._build_static_feature），
    # 默认 True = 「签名与位图一起」。
    USE_BITMAP = bool(config.get('use_bitmap', True))
    # DropSegment 的 m：论文 §3 的**固定上限**（每次删 r ~ U{0..m} 段），不是百分比
    DROP_MAX_SEGMENTS = int(config.get('drop_max_segments', 1))
    # 评估批量与 worker 数**默认与训练对齐**，只有显式配置时才单独设。
    # 早先评估写死 min(8, NUM_WORKERS) + batch 128，比训练（16 worker / 192）
    # 慢一截，还漏了 worker_init_fn（20-test 的随机性会因此退化）。
    EVAL_BATCH_SIZE = int(config.get('eval_batch_size', BATCH_SIZE))
    EVAL_NUM_WORKERS = int(config.get('eval_num_workers', NUM_WORKERS))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    any_dynamic = any(r.needs_dynamic for r in runs)

    # ---- 阶段解析 ----
    for r in runs:
        r.stages = resolve_run_stages(r, config)
        build_run_schedule(r)
    max_epochs = max(r.total_epochs for r in runs)

    print("=" * 80)
    if len(runs) == 1:
        print(f"🚀 单模型训练 | {runs[0].name} | 设备 = {DEVICE}")
    else:
        print(f"🚀 多模型共享管道训练 | {len(runs)} 个模型 | 设备 = {DEVICE}")
    print(f"   batch_size={BATCH_SIZE}  num_workers={NUM_WORKERS}  "
          f"总轮数={max_epochs}")
    print(f"   静态特征: 1 位图 + 63 签名（{STATIC_FEATURE_VERSION}，按原文实现）")
    print(f"   动态特征: {'启用（有模型需要）' if any_dynamic else '关闭（都不需要，省时间）'}")
    print("-" * 80)
    for r in runs:
        print(f"   · {r.name:14s} arch={r.arch:9s} mode={r.mode:8s} "
              f"opt={r.optimizer_name:6s} lr={r.lr:.2e} wd={r.weight_decay:g} "
              f"轮数={r.total_epochs}")
        if len(r.stages) > 1 or r.raw.get('stages'):
            for st in r.stages:
                print(f"       └ {st['name']:24s} epochs={st['epochs']:<3} "
                      f"lr={st['lr']:.2e} freeze_static={st['freeze_static']} "
                      f"static_lr_scale={st['static_lr_scale']}")
    if len(runs) > 1:
        print("-" * 80)
        print("ℹ️  所有模型共享同一份内存数据与同一个 DataLoader ——")
        print("   特征提取每个 batch 只做一次，且各模型看到的增强样本完全一致。")
        if len({r.total_epochs for r in runs}) > 1:
            print("⚠️  各模型总轮数不同：轮数少的跑完后会**停止参与训练**，"
                  "但仍保留在断点中。")
    print("=" * 80)

    # ========== 数据 ==========
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metadata_file = config.get('metadata_file', 'data/features/metadata.csv')
    if not os.path.isabs(metadata_file):
        metadata_file = os.path.join(BASE_DIR, metadata_file)

    full_df = pd.read_csv(metadata_file, encoding='utf-8-sig')
    full_df['writer_id'] = full_df['writer_id'].astype(str)
    all_writer_ids = sorted(full_df['writer_id'].unique())
    global_label_map = {wid: i for i, wid in enumerate(all_writer_ids)}
    num_classes = len(global_label_map)
    print(f"📊 类别数: {num_classes}")

    # 几何增强已移到**坐标域**（见 src/data/dataset.py::_augment_points）：
    # 论文要求在「算签名特征图之前」做平移+旋转+缩放，签名才会随之改变。
    # 早期实现用 RandomRotation/RandomAffine 作用在渲染好的 64 通道图上，
    # 签名数值其实没变（只是同一张图被转过头），且缺缩放。
    train_transform = transforms.Compose([_pad_to_96])

    train_dataset = WPTTDataset(
        metadata_file, split='Train', transform=train_transform,
        page_workers=config.get('page_workers', 8),
        segment_workers=config.get('segment_workers', 8),
        apply_drop=config.get('apply_drop', True),
        return_dynamic=any_dynamic,
        seq_len=SEQ_LEN,
        dynamic_speed_clip=DYNAMIC_SPEED_CLIP,
        augment_points=AUGMENT_POINTS,
        use_bitmap=USE_BITMAP,
        drop_max_segments=DROP_MAX_SEGMENTS,
    )
    print(f"📊 训练集样本数: {len(train_dataset)}（内存中只此一份）")

    # ========== 模型 ==========
    model_cfg = dict(config.get('model', {}) or {})
    pretrained_path = model_cfg.get('pretrained_path')
    if pretrained_path and not os.path.isabs(pretrained_path):
        pretrained_path = os.path.join(BASE_DIR, pretrained_path)

    for r in runs:
        per_run = dict(model_cfg)
        per_run.update(r.extra)
        per_run['arch'] = r.arch
        per_run['mode'] = r.mode
        r.model = build_model(per_run, num_classes,
                              pretrained_path=pretrained_path).to(DEVICE)
        if r.static_init_path and r.mode in ('static', 'dual'):
            r.model.load_static_weights(r.static_init_path)
        print(f"🧠 [{r.name}] 参数量 {r.model.num_total_params()/1e6:.3f}M "
              f"（可训练 {r.model.num_trainable_params()/1e6:.3f}M）")

    criterion = nn.CrossEntropyLoss()

    # ========== 路径 ==========
    ckpt_cfg = config.get('checkpoint', {}) or {}
    log_cfg = config.get('logging', {}) or {}
    ckpt_dir = ckpt_cfg.get('save_dir', 'outputs/checkpoints')
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(BASE_DIR, ckpt_dir)
    _ensure_dir(ckpt_dir, "检查点")

    log_dir = log_cfg.get('log_dir', 'outputs/logs')
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(BASE_DIR, log_dir)
    log_path = os.path.join(log_dir, log_cfg.get('log_file', 'training_log.csv'))

    group = ckpt_cfg.get('group') or (
        runs[0].prefix if len(runs) == 1 else 'multi_arch')
    combined_path = os.path.join(ckpt_dir, f'{group}_latest.pth')
    config_path = _dump_config(config, runs, ckpt_dir, group)

    steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)

    # ========== 断点续训 ==========
    # 关键：必须先判断「能不能续」，再决定日志是清空还是接续。
    # 之前把 _init_log 放在这里之前且无条件 append，全新训练会把本轮数据
    # 追加到上一次实验的旧行后面，两个实验混进同一个 CSV。
    start_epoch = 1
    resume_state = None
    resume_why = None

    if not config.get('resume', True):
        resume_why = "--no_resume / resume=false"
    elif not os.path.exists(combined_path):
        resume_why = "未找到检查点"
    else:
        ckpt = torch.load(combined_path, map_location=DEVICE, weights_only=False)
        saved_cfg = ckpt.get('config', {}) or {}
        problems = []

        if ckpt.get('run_names') != [r.name for r in runs]:
            problems.append(
                f"模型集合不同: 断点={ckpt.get('run_names')} 当前={[r.name for r in runs]}")

        # ★ 特征编码版本必须一致。
        #   早期还有 separator_aware / fix_signature_layout 两个开关可以切到
        #   v2.0 的缺陷行为（A0），现在那条路径已整个删除，只剩这一个版本号。
        #   版本号覆盖了「同一条算法内部改过」的情况 —— 例如签名加了尺度归一化
        #   和 4×4 膨胀，开关名字不变但 64 通道的语义已经不同，此时续训会把
        #   旧权重接进新特征，准确率莫名其妙地低，只看开关是查不出来的。
        old_ver = saved_cfg.get('static_feature_version')
        if old_ver is not None and old_ver != STATIC_FEATURE_VERSION:
            problems.append(
                f"特征编码版本不一致: 断点 {old_ver} ≠ 当前 "
                f"{STATIC_FEATURE_VERSION}（特征算法改过，旧权重不可复用）")

        # 各 run 的架构 / 模式 / 优化器也不能变
        saved_runs = {x['name']: x for x in (saved_cfg.get('_runs') or [])}
        for r in runs:
            old = saved_runs.get(r.name)
            if old is None:
                continue
            if old.get('arch') != r.arch:
                problems.append(f"[{r.name}] arch: 断点 {old.get('arch')} ≠ 当前 {r.arch}")
            if old.get('mode') != r.mode:
                problems.append(f"[{r.name}] mode: 断点 {old.get('mode')} ≠ 当前 {r.mode}")
            if old.get('optimizer') != r.optimizer_name:
                problems.append(f"[{r.name}] optimizer: 断点 {old.get('optimizer')} "
                                f"≠ 当前 {r.optimizer_name}")

        if problems:
            resume_why = "检查点与当前配置不兼容"
            print("⚠️  " + "=" * 74)
            print("⚠️  拒绝从该检查点续训，改为从零开始。原因：")
            for p in problems:
                print(f"⚠️     · {p}")
            print(f"⚠️  断点文件: {combined_path}")
            print("⚠️  如需保留它，请先改名/移动；确认要丢弃可直接忽略。")
            print("⚠️  " + "=" * 74)
        else:
            for r in runs:
                r.model.load_state_dict(ckpt['models'][r.name], strict=False)
                r.best_acc = float(ckpt.get('best', {}).get(r.name, 0.0))
                r.best_epoch = int((ckpt.get('best_epoch') or {}).get(r.name, 0))
            start_epoch = int(ckpt.get('epoch', 0)) + 1
            resume_state = ckpt
            print(f"✅ 断点恢复: 从第 {start_epoch} 轮继续  "
                  f"({', '.join(f'{r.name}={r.best_acc:.4f}' for r in runs)})")

    fresh = resume_state is None
    if fresh:
        print(f"🆕 全新训练，从第 1 轮开始（原因: {resume_why}）")
    _init_log(log_path, runs, reset=fresh)

    if start_epoch > max_epochs:
        print(f"🏁 已训练到 {max_epochs} 轮，无需继续。")
        summary_path = os.path.join(os.path.dirname(log_path),
                                    f'{group}_summary.csv')
        _dump_summary(summary_path, runs, ckpt_dir,
                      time.time() - train_start, max_epochs)
        return {'runs': {r.name: {'best_acc': r.best_acc,
                                  'best_epoch': r.best_epoch,
                                  'best_path': r.best_path(ckpt_dir),
                                  'arch': r.arch, 'mode': r.mode}
                         for r in runs},
                'combined_path': combined_path, 'log_path': log_path,
                'config_path': config_path, 'summary_path': summary_path,
                'wall_time_sec': time.time() - train_start,
                'epochs': max_epochs}

    # ========== 训练循环 ==========
    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        # ---- 阶段切换：需要时重建该 run 的优化器/调度器 ----
        active = []
        for r in runs:
            if epoch > r.total_epochs:
                continue                      # 该模型已跑完自己的轮数
            si, ei = r.schedule[epoch]
            stage = r.stages[si]
            if r.cur_stage != si:
                r.cur_stage = si
                r.optimizer = build_optimizer(r, stage)
                r.scheduler = build_scheduler(r, stage, steps_per_epoch)
                note = (f"（静态分支冻结={r.model.static_frozen}）"
                        if getattr(r.model, 'static_branch', None) is not None
                        else "")
                print(f"\n🎬 [{r.name}] 进入阶段 {si+1}/{len(r.stages)}: "
                      f"{stage['name']}  lr={float(stage['lr']):.2e} {note}")
                if (resume_state is not None and epoch == start_epoch
                        and r.name in (resume_state.get('optimizers') or {})):
                    try:
                        r.optimizer.load_state_dict(
                            resume_state['optimizers'][r.name])
                        r.scheduler.load_state_dict(
                            resume_state['schedulers'][r.name])
                        print(f"   ✅ [{r.name}] 已恢复优化器/调度器状态")
                    except Exception as exc:              # noqa: BLE001
                        print(f"   ⚠️ [{r.name}] 优化器状态恢复失败: {exc}")
            active.append((r, si, ei))
        resume_state = None

        loader = _make_loader(train_dataset, BATCH_SIZE, NUM_WORKERS,
                              shuffle=True)

        for r, _, _ in active:
            r.model.train()

        stats = {r.name: {'loss': 0.0, 'correct': 0, 'correct_top5': 0,
                          'total': 0, 'grad_norm': 0.0, 'gn_steps': 0}
                 for r, _, _ in active}
        compute_time = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{max_epochs}")

        for batch in pbar:
            inputs, dynamic, labels = _unpack_batch(batch)
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            dyn = (dynamic.to(DEVICE, non_blocking=True)
                   if dynamic is not None else None)

            t_compute = time.perf_counter()
            postfix = {}
            # ---- 同一个 batch 依次喂给每个活跃模型 ----
            # 每个模型 backward 完立刻释放自己的计算图 → 显存峰值 ≈ max 而非求和
            for r, _, _ in active:
                r.optimizer.zero_grad(set_to_none=True)
                out = r.model(inputs, dyn) if r.needs_dynamic else r.model(inputs)
                loss = criterion(out, labels)
                loss.backward()

                # clip_grad_norm_ 返回**裁剪前**的范数，顺手记下来
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in r.model.parameters() if p.requires_grad],
                    GRAD_CLIP if GRAD_CLIP else float('inf'))
                r.optimizer.step()
                r.scheduler.step()

                s = stats[r.name]
                s['loss'] += loss.item() * inputs.size(0)
                s['correct'] += (out.argmax(1) == labels).sum().item()
                if out.size(1) >= 5:
                    top5 = out.topk(5, dim=1).indices
                    s['correct_top5'] += (top5 == labels[:, None]).any(1).sum().item()
                else:
                    s['correct_top5'] += (out.argmax(1) == labels).sum().item()
                s['total'] += labels.size(0)
                s['grad_norm'] += float(gn)
                s['gn_steps'] += 1
                postfix[r.name] = (f"{loss.item():.3f}/"
                                   f"{s['correct']/max(s['total'],1):.3f}")

            compute_time += time.perf_counter() - t_compute
            pbar.set_postfix(postfix, refresh=False)

        del loader
        gc.collect()

        epoch_time = time.time() - t0
        # 近似拆分：数据管道与计算是重叠的，所以「数据时间」≈ 等数据的空闲时间
        data_time = max(epoch_time - compute_time, 0.0)
        gpu_peak = (torch.cuda.max_memory_allocated() / 1e6
                    if torch.cuda.is_available() else 0.0)

        # ---- 本轮统计 ----
        train_stats = {}
        print(f"\n{'='*80}\nEpoch {epoch}/{max_epochs} 训练结果")
        for r, si, ei in active:
            s = stats[r.name]
            loss = s['loss'] / max(s['total'], 1)
            acc = s['correct'] / max(s['total'], 1)
            acc5 = s['correct_top5'] / max(s['total'], 1)
            gn = s['grad_norm'] / max(s['gn_steps'], 1)
            train_stats[r.name] = {'loss': loss, 'char_acc': acc,
                                   'char_top5': acc5, 'grad_norm': gn,
                                   'si': si, 'ei': ei}
            print(f"   {r.name:14s} [阶段{si+1} {ei}/{r.stages[si]['epochs']}] "
                  f"loss={loss:.4f}  char_acc={acc:.4f}  top5={acc5:.4f}  "
                  f"grad_norm={gn:.3f}")

        # ---- 评估（两个模型共享同一份测试集，只建一次）----
        eval_details = {}
        to_eval = [r for r, _, _ in active if epoch % r.eval_frequency == 0]
        if to_eval:
            dets = evaluate_page_level_multi(
                {r.name: r.model for r in to_eval},
                metadata_file, global_label_map, DEVICE,
                batch_size=EVAL_BATCH_SIZE,
                num_workers=EVAL_NUM_WORKERS,
                page_workers=config.get('page_workers', 8),
                segment_workers=config.get('segment_workers', 8),
                apply_drop=False,
                needs_dynamic={r.name: r.needs_dynamic for r in to_eval},
                seq_len=SEQ_LEN,
                dynamic_speed_clip=DYNAMIC_SPEED_CLIP,
                use_bitmap=USE_BITMAP,
            )
            for r in to_eval:
                det = dets[r.name]
                eval_details[r.name] = det
                acc = det['page_acc']
                print(f"   {r.name:14s} page_acc={acc:.4f}  "
                      f"page_top5={det['page_acc_top5']:.4f}")

                if acc > r.best_acc:
                    r.best_acc = acc
                    r.best_epoch = epoch
                    _save_single(r, ckpt_dir, epoch, config)
                    print(f"      🎉 新最佳 → {r.best_path(ckpt_dir)}")

        # ---- 组合检查点（断点续训用） ----
        torch.save({
            'epoch': epoch,
            'run_names': [r.name for r in runs],
            'models': {r.name: r.model.state_dict() for r in runs},
            'optimizers': {r.name: r.optimizer.state_dict()
                           for r in runs if r.optimizer is not None},
            'schedulers': {r.name: r.scheduler.state_dict()
                           for r in runs if r.scheduler is not None},
            'best': {r.name: r.best_acc for r in runs},
            'best_epoch': {r.name: r.best_epoch for r in runs},
            'config': {
                **{k: v for k, v in config.items() if k != 'runs'},
                'static_feature_version': STATIC_FEATURE_VERSION,
                # 每个 run 的身份信息，续训时用来逐项校验兼容性
                '_runs': [
                    {'name': r.name, 'arch': r.arch, 'mode': r.mode,
                     'prefix': r.prefix, 'optimizer': r.optimizer_name,
                     'lr': r.lr, 'total_epochs': r.total_epochs}
                    for r in runs
                ],
            },
        }, combined_path)

        # ---- 写逐轮 CSV：训练过程产生的所有指标 ----
        row = [epoch, time.strftime('%Y-%m-%d %H:%M:%S'),
               f"{epoch_time:.1f}", f"{compute_time:.1f}", f"{data_time:.1f}"]
        for r in runs:
            ts = train_stats.get(r.name)
            if ts is None:                      # 该模型本轮未参与
                row += ['-'] + [''] * (len(PER_RUN_COLUMNS) - 1)
                continue
            ev = eval_details.get(r.name, {})
            si, ei = ts['si'], ts['ei']
            lrs = [g['lr'] for g in r.optimizer.param_groups]
            lr_main = lrs[-1]
            lr_static = (lrs[0] if len(lrs) > 1 else '')
            row += [
                f"{si+1}.{ei}/{r.stages[si]['epochs']}",
                f"{ts['loss']:.6f}", f"{ts['char_acc']:.6f}",
                f"{ts['char_top5']:.6f}",
                f"{ev.get('char_acc', float('nan')):.6f}" if ev else "",
                f"{ev.get('char_acc_top5', float('nan')):.6f}" if ev else "",
                f"{ev.get('page_acc', float('nan')):.6f}" if ev else "",
                f"{ev.get('page_acc_top5', float('nan')):.6f}" if ev else "",
                f"{r.best_acc:.6f}",
                1 if ev and ev.get('page_acc', 0) >= r.best_acc else 0,
                f"{lr_main:.8f}",
                f"{lr_static:.8f}" if lr_static != '' else "",
                f"{ts['grad_norm']:.4f}",
                f"{gpu_peak:.1f}",
            ]
        _append_log(log_path, row)

    # ========== 结束 ==========
    total_wall = time.time() - train_start
    summary_path = os.path.join(os.path.dirname(log_path),
                                f'{group}_summary.csv')
    _dump_summary(summary_path, runs, ckpt_dir, total_wall, max_epochs)

    print("\n" + "=" * 80)
    print(f"🏁 训练完成（总耗时 {total_wall/3600:.2f} 小时）")
    for r in runs:
        print(f"   {r.name:14s} best_page_acc = {r.best_acc:.4f} "
              f"@ epoch {r.best_epoch}  → {r.best_path(ckpt_dir)}")
    print(f"   组合检查点（可续训）: {combined_path}")
    print(f"   逐轮日志            : {log_path}")
    print(f"   配置与环境存档      : {config_path}")
    print(f"   训练小结            : {summary_path}")
    print("=" * 80)

    # ========== 20-test 增强评估（训练结束后，用各模型的最佳权重）==========
    t20 = config.get('eval_20test') or {}
    test20 = None
    if t20.get('enabled', True) and int(t20.get('n_test', 20)) > 0:
        n_test = int(t20.get('n_test', 20))
        print("\n" + "=" * 80)
        print(f"🔁 开始 {n_test}-test 增强评估（所有模型共享同一批 DropSegment 增强）")
        print("   用每个模型的**最佳权重**（而非最后一轮），与论文口径一致。")
        print("=" * 80)

        # 把内存里的模型换成 best 权重，否则评的是最后一轮
        best_models, needs_dyn = {}, {}
        for r in runs:
            bp = r.best_path(ckpt_dir)
            if os.path.exists(bp):
                ck = torch.load(bp, map_location=DEVICE, weights_only=False)
                r.model.load_state_dict(ck['model_state_dict'])
                print(f"   ↩ [{r.name}] 已载入最佳权重 (epoch {ck.get('epoch')}, "
                      f"page_acc={ck.get('best_acc', 0):.4f})")
            else:
                print(f"   ⚠️ [{r.name}] 找不到 {os.path.basename(bp)}，"
                      f"用当前（最后一轮）权重")
            best_models[r.name] = r.model
            needs_dyn[r.name] = r.needs_dynamic

        try:
            test20 = evaluate_page_level_20test_multi(
                best_models, metadata_file, global_label_map, DEVICE,
                n_test=n_test,
                batch_size=int(t20.get('batch_size', EVAL_BATCH_SIZE)),
                num_workers=int(t20.get('num_workers', EVAL_NUM_WORKERS)),
                page_workers=config.get('page_workers', 8),
                segment_workers=config.get('segment_workers', 8),
                apply_drop=True,
                needs_dynamic=needs_dyn,
                seq_len=SEQ_LEN,
                dynamic_speed_clip=DYNAMIC_SPEED_CLIP,
                use_bitmap=USE_BITMAP,
                log_path=os.path.join(log_dir, f'{group}_20test_records.csv'),
            )
        except Exception as exc:                       # noqa: BLE001
            print(f"⚠️ 20-test 失败（不影响训练产物）: {exc}")
            import traceback
            traceback.print_exc()

    return {'runs': {r.name: {'best_acc': r.best_acc,
                              'best_epoch': r.best_epoch,
                              'best_path': r.best_path(ckpt_dir),
                              'arch': r.arch, 'mode': r.mode} for r in runs},
            'combined_path': combined_path, 'log_path': log_path,
            'config_path': config_path, 'summary_path': summary_path,
            'test20': test20,
            'wall_time_sec': total_wall,
            'epochs': max_epochs}


def _save_single(run: RunSpec, ckpt_dir: str, epoch: int, config: dict):
    """
    按**标准格式**保存单个模型的最佳权重。

    刻意与 train_dual.py 的 checkpoint 格式一致（含 'config'），这样
    analyze_complementarity.py / evaluate_v2.py 能直接读，
    也保留了静态流水线变体记录（热启动护栏依赖它）。
    """
    per_run_cfg = dict(config.get('model', {}) or {})
    per_run_cfg.update(run.extra)
    per_run_cfg['arch'] = run.arch
    per_run_cfg['mode'] = run.mode
    per_run_cfg['feature_dim'] = per_run_cfg.get('feature_dim', 512)
    # 特征编码版本：让下游脚本（eval_20test / analyze）能识别权重是否过期
    per_run_cfg['static_feature_version'] = STATIC_FEATURE_VERSION

    torch.save({
        'epoch': epoch,
        'model_state_dict': run.model.state_dict(),
        'best_acc': run.best_acc,
        'arch': run.arch,
        'mode': run.mode,
        'run_name': run.name,
        'config': per_run_cfg,
    }, run.best_path(ckpt_dir))
