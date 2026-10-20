"""
双路融合三阶段训练器。

三种模式（消融实验 A/B/C）共用这一个训练器，只由 config 的 `mode` 区分：
    static  → 实验 A：纯静态基线（数值等价于原 ConvNeXtBackbone）
    dynamic → 实验 B：纯动态分支
    dual    → 实验 C：双路融合（本方案主角）

三阶段训练（仅 dual 模式有意义）
--------------------------------
| 阶段 | 静态分支   | 动态分支 | 融合层 | 学习率 | 轮数 |
| 1    | 冻结       | 训练     | 训练   | 1e-3   | 5    |
| 2    | 冻结       | 训练     | 训练   | 5e-4   | 20   |
| 3    | 微调(小lr) | 微调     | 训练   | 1e-4   | 25   |

为什么分阶段：
- 阶段 1：动态分支随机初始化、梯度很大。若同时解冻静态分支，
  这些大梯度会冲坏已经收敛的预训练权重。
- 阶段 3：动态分支稳定后联合微调，让静态特征去适配动态特征，
  才能拿到融合增益。

工程要点
--------
- 每个 epoch 重建 train_loader（继承原 trainer 的设计，避免
  worker 与测试阶段的多进程冲突）。
- 每个阶段重建优化器：阶段切换时静态分支的 requires_grad 会变化，
  参数集合不同，必须重建（调度器同理）。
- 断点续训记录 (stage_idx, epoch_in_stage)，可精确恢复到阶段内某轮。
- 阶段结束额外保存 `*_stage{k}.pth`，云端跑挂也不至于全盘丢失。
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
from src.evaluation.evaluator import evaluate_page_level

_np = np

#: Linux/云端可用 fork（省内存、启动快）；Windows 只有 spawn，传 'fork' 会直接报错。
MP_CONTEXT = 'fork' if sys.platform != 'win32' else None


def _loader_context(num_workers):
    """num_workers>0 时才指定多进程上下文，否则必须为 None。"""
    return MP_CONTEXT if num_workers > 0 else None


def _worker_init_fn(worker_id):
    """每个 DataLoader worker 独立随机种子，保证 DropSegment 真随机。"""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    _random.seed(worker_seed)
    _np.random.seed(worker_seed)


def _pad_to_96(tensor):
    """v1.0 继承：54×54 → 96×96。"""
    pad = (21, 21, 21, 21)
    return torch.nn.functional.pad(tensor, pad, mode='constant', value=0)


def _unpack_batch(batch):
    """兼容 (静态, 标签) 与 (静态, 动态, 标签) 两种批次结构。"""
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    if len(batch) == 2:
        return batch[0], None, batch[1]
    raise ValueError(f"无法识别的批次结构，元素个数={len(batch)}")


# ----------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------

_LOG_HEADER = ['stage', 'stage_name', 'epoch', 'global_epoch', 'train_loss',
               'train_char_acc', 'test_page_acc', 'lr', 'epoch_time_sec']


def _ensure_dir(path, what):
    """创建输出目录；失败时给出可操作的提示而不是原始 traceback。"""
    try:
        os.makedirs(path, exist_ok=True)
    except (PermissionError, OSError) as exc:
        raise PermissionError(
            f"无法创建{what}目录: {path}\n"
            f"  原因: {type(exc).__name__}: {exc}\n"
            f"  处理办法（任选其一）:\n"
            f"    1. 用 --set checkpoint.save_dir=<可写路径> "
            f"logging.log_dir=<可写路径> 换一个目录；\n"
            f"    2. 把整个项目目录的写权限授予当前用户后重试；\n"
            f"    3. 在 Linux / 云端运行（推荐，本仓库的主要运行环境）。"
        ) from exc
    return path


def _init_log_file(log_path):
    _ensure_dir(os.path.dirname(log_path), "训练日志")
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(_LOG_HEADER)
        print(f"📝 已创建训练日志: {log_path}")


def _append_log(log_path, row):
    with open(log_path, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(row)


# ----------------------------------------------------------------------
# 阶段配置
# ----------------------------------------------------------------------

DEFAULT_STAGES = {
    # 双路融合：三阶段
    'dual': [
        {'name': 'stage1_head_warmup', 'epochs': 5, 'lr': 1e-3,
         'freeze_static': True, 'static_lr_scale': 0.0},
        {'name': 'stage2_head_converge', 'epochs': 20, 'lr': 5e-4,
         'freeze_static': True, 'static_lr_scale': 0.0},
        {'name': 'stage3_joint_finetune', 'epochs': 25, 'lr': 1e-4,
         'freeze_static': False, 'static_lr_scale': 0.1},
    ],
    # 纯动态：没有静态分支可冻结，前两阶段照跑，第三阶段降 lr 续训
    'dynamic': [
        {'name': 'stage1_warmup', 'epochs': 5, 'lr': 1e-3,
         'freeze_static': True, 'static_lr_scale': 0.0},
        {'name': 'stage2_converge', 'epochs': 20, 'lr': 5e-4,
         'freeze_static': True, 'static_lr_scale': 0.0},
        {'name': 'stage3_finetune', 'epochs': 25, 'lr': 1e-4,
         'freeze_static': True, 'static_lr_scale': 0.0},
    ],
    # 纯静态：单阶段，直接复现原基线训练
    # lr=1.0 是占位值，实际学习率 = 1.0 × (lr_base × batch/256)，与原 trainer 对齐
    'static': [
        {'name': 'static_baseline', 'epochs': 50, 'lr': 1.0,
         'freeze_static': False, 'static_lr_scale': 1.0},
    ],
}


def resolve_stages(config):
    """
    把 config 里的训练阶段规范化成带默认值的列表。

    优先级：
        1. config['stages_by_mode'][mode]  —— 推荐，按模式分开配置
        2. config['stages']                —— 显式覆盖（--set stages=... 用）
        3. DEFAULT_STAGES[mode]            —— 代码内置兜底

    为什么必须按模式分开：纯静态基线要用「单阶段 + 对齐 v1.0 的学习率」，
    若误用双路的三阶段计划，stage lr 会被 lr_base 缩放成 5e-7，
    等于几乎不训练，实验 A 会得出错误结论。
    """
    mode = config.get('mode', 'dual')

    by_mode = config.get('stages_by_mode') or {}
    stages = by_mode.get(mode)
    if stages is None:
        stages = config.get('stages')
    if stages is None:
        stages = DEFAULT_STAGES.get(mode)
    if not stages:
        raise ValueError(f"模式 {mode} 没有可用的训练阶段配置")

    out = []
    for i, st in enumerate(stages):
        st = dict(st)
        st.setdefault('name', f'stage{i+1}')
        st.setdefault('freeze_static', mode == 'dual' and i < len(stages) - 1)
        st.setdefault('static_lr_scale', 0.1 if mode == 'dual' else 1.0)
        if st.get('lr') is None:
            raise ValueError(f"阶段 {st['name']} 缺少 lr")
        out.append(st)
    return out


# ----------------------------------------------------------------------
# 优化器 / 调度器
# ----------------------------------------------------------------------

def build_optimizer(model, stage, weight_decay, base_lr_scale=1.0):
    """
    按阶段构造优化器。

    阶段 3 解冻静态分支时，静态分支用更小的学习率（static_lr_scale），
    避免把已经收敛的预训练权重整体带偏。
    """
    lr = float(stage['lr']) * base_lr_scale
    static_scale = float(stage.get('static_lr_scale', 1.0))

    freeze_static = bool(stage.get('freeze_static', False))
    model.set_static_frozen(freeze_static)

    static_ids = set()
    if model.static_branch is not None:
        static_ids = {id(p) for p in model.static_branch.parameters()}

    static_params, other_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (static_params if id(p) in static_ids else other_params).append(p)

    groups = []
    if static_params:
        groups.append({'params': static_params, 'lr': lr * static_scale,
                       'name': 'static'})
    if other_params:
        groups.append({'params': other_params, 'lr': lr, 'name': 'dynamic_fusion'})
    if not groups:
        raise RuntimeError("没有可训练参数，请检查 freeze_static 设置")

    optimizer = optim.AdamW(groups, lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.999))
    return optimizer, lr


def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs,
                    base_lr):
    """Warmup + 余弦退火，iteration 级 step。"""
    warmup_steps = max(int(warmup_epochs * steps_per_epoch), 1)
    cosine_steps = max(int((epochs - warmup_epochs) * steps_per_epoch), 1)
    cos_min = max(base_lr * 0.01, 1e-6)

    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=cos_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_steps])


# ----------------------------------------------------------------------
# 主训练
# ----------------------------------------------------------------------

def train_dual(config):
    mode = config.get('mode', 'dual')
    BATCH_SIZE = config.get('batch_size', 192)
    WEIGHT_DECAY = config.get('weight_decay', 0.05)
    WARMUP_EPOCHS = config.get('warmup_epochs', 1)
    EVAL_FREQUENCY = config.get('eval_frequency', 1)
    PATIENCE = config.get('patience', 10)
    NUM_WORKERS = config.get('num_workers', 16)
    SEQ_LEN = config.get('seq_len', 200)
    DYNAMIC_SPEED_CLIP = config.get('dynamic_speed_clip', 10.0)

    # 纯静态模式下 LR 沿用 lr_base × batch/256 的 v1.0 对齐规则
    LR_BASE = config.get('lr_base', 0.000667)
    BASE_LR_SCALE = LR_BASE * BATCH_SIZE / 256 if mode == 'static' else 1.0

    stages = resolve_stages(config)
    return_dynamic = mode in ('dynamic', 'dual')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 74)
    print(f"🚀 双路融合训练 | 模式 = {mode} | 设备 = {DEVICE}")
    print(f"   batch={BATCH_SIZE}  workers={NUM_WORKERS}  seq_len={SEQ_LEN}  "
          f"dynamic_speed_clip={DYNAMIC_SPEED_CLIP}")
    print(f"   return_dynamic={return_dynamic}  (动态特征在 DropSegment 之后提取)")
    print(f"   静态特征: 1 位图 + 63 签名（{STATIC_FEATURE_VERSION}，按原文实现）")
    for i, st in enumerate(stages):
        print(f"   阶段{i+1}: {st['name']:<24} epochs={st['epochs']:<3} "
              f"lr={float(st['lr']) * BASE_LR_SCALE:.2e} "
              f"freeze_static={st.get('freeze_static')} "
              f"static_lr_scale={st.get('static_lr_scale')}")
    print("=" * 74)

    # ========== 数据路径 ==========
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metadata_file = config.get('metadata_file',
                               os.path.join(BASE_DIR, 'data', 'features', 'metadata.csv'))
    if not os.path.isabs(metadata_file):
        metadata_file = os.path.join(BASE_DIR, metadata_file)

    full_df = pd.read_csv(metadata_file, encoding='utf-8-sig')
    full_df['writer_id'] = full_df['writer_id'].astype(str)
    all_writer_ids = sorted(full_df['writer_id'].unique())
    global_label_map = {wid: idx for idx, wid in enumerate(all_writer_ids)}
    num_classes = len(global_label_map)
    print(f"📊 类别数: {num_classes}")

    # ========== 数据增强 ==========
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        _pad_to_96,
    ])

    train_dataset = WPTTDataset(
        metadata_file, split='Train', transform=train_transform,
        page_workers=config.get('page_workers', 8),
        segment_workers=config.get('segment_workers', 8),
        apply_drop=config.get('apply_drop', True),
        return_dynamic=return_dynamic,
        seq_len=SEQ_LEN,
        dynamic_speed_clip=DYNAMIC_SPEED_CLIP,
    )
    print(f"📊 训练集样本数: {len(train_dataset)}")

    # ========== 模型 ==========
    pretrained_path = config.get('pretrained_path')
    if pretrained_path and not os.path.isabs(pretrained_path):
        pretrained_path = os.path.join(BASE_DIR, pretrained_path)

    model = build_model(config, num_classes, pretrained_path=pretrained_path)
    model = model.to(DEVICE)
    print(f"🧠 参数量: 总计 {model.num_total_params()/1e6:.2f}M, "
          f"可训练 {model.num_trainable_params()/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()

    # ========== 输出路径 ==========
    ckpt_cfg = config.get('checkpoint', {}) or {}
    log_cfg = config.get('logging', {}) or {}
    prefix = ckpt_cfg.get('prefix', f'{mode}_stream')
    ckpt_dir = ckpt_cfg.get('save_dir', 'outputs/checkpoints')
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(BASE_DIR, ckpt_dir)
    _ensure_dir(ckpt_dir, "检查点")

    latest_path = os.path.join(ckpt_dir, f'{prefix}_latest.pth')
    best_path = os.path.join(ckpt_dir, f'{prefix}_best.pth')

    log_dir = log_cfg.get('log_dir', 'outputs/logs')
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(BASE_DIR, log_dir)
    log_path = os.path.join(log_dir, log_cfg.get('log_file', f'{prefix}_training_log.csv'))
    _init_log_file(log_path)

    # ========== 断点续训 ==========
    start_stage_idx, start_epoch_in_stage = 0, 1
    best_page_acc = 0.0
    global_epoch = 0
    resume_state = None          # 暂存，用于在对应阶段恢复优化器/调度器状态

    resume = config.get('resume', True)
    if resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_stage_idx = int(ckpt.get('stage_idx', 0))
        start_epoch_in_stage = int(ckpt.get('epoch_in_stage', 0)) + 1
        best_page_acc = float(ckpt.get('best_acc', 0.0))
        global_epoch = int(ckpt.get('global_epoch', 0))
        resume_state = ckpt
        print(f"✅ 断点恢复: 阶段 {start_stage_idx+1}, 阶段内第 {start_epoch_in_stage} 轮, "
              f"Best Page Acc: {best_page_acc:.4f}")

        # 阶段已跑完 → 进入下一阶段第一轮
        while (start_stage_idx < len(stages)
               and start_epoch_in_stage > int(stages[start_stage_idx]['epochs'])):
            start_stage_idx += 1
            start_epoch_in_stage = 1
            resume_state = None      # 跨阶段了，旧阶段的优化器状态不再适用
    else:
        print("🆕 未找到检查点，将从阶段 1 第 1 轮开始训练。")

    def _save(path, stage_idx, epoch_in_stage, optimizer=None, scheduler=None):
        payload = {
            'mode': mode,
            'stage_idx': stage_idx,
            'epoch_in_stage': epoch_in_stage,
            'global_epoch': global_epoch,
            'best_acc': best_page_acc,
            'model_state_dict': model.state_dict(),
            'config': {k: v for k, v in config.items() if k != 'stages'},
        }
        if optimizer is not None:
            payload['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            payload['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(payload, path)

    # ========== 阶段主循环 ==========
    for stage_idx in range(start_stage_idx, len(stages)):
        stage = stages[stage_idx]
        stage_epochs = int(stage['epochs'])
        first_epoch = start_epoch_in_stage if stage_idx == start_stage_idx else 1

        if first_epoch > stage_epochs:
            continue

        print("\n" + "=" * 74)
        print(f"🎬 阶段 {stage_idx+1}/{len(stages)}: {stage['name']}  "
              f"(epochs {first_epoch}→{stage_epochs})")
        print("=" * 74)

        steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
        optimizer, stage_lr = build_optimizer(model, stage, WEIGHT_DECAY,
                                              BASE_LR_SCALE)
        scheduler = build_scheduler(optimizer, stage_epochs, steps_per_epoch,
                                    WARMUP_EPOCHS, stage_lr)

        # 恢复同一阶段内中断时的优化器/调度器状态（含 AdamW 动量），
        # 避免续训后动量清零导致的短暂性能抖动
        if (resume_state is not None and stage_idx == start_stage_idx
                and first_epoch > 1):
            if 'optimizer_state_dict' in resume_state:
                try:
                    optimizer.load_state_dict(resume_state['optimizer_state_dict'])
                    scheduler.load_state_dict(resume_state['scheduler_state_dict'])
                    print(f"   ✅ 已恢复优化器/调度器状态"
                          f"（阶段内第 {first_epoch} 轮继续）")
                except Exception as exc:              # noqa: BLE001
                    print(f"   ⚠️ 优化器状态恢复失败，将重新初始化: {exc}")
            else:
                print("   ⚠️ 检查点中没有优化器状态，优化器将重新初始化")
            resume_state = None

        print(f"   可训练参数: {model.num_trainable_params()/1e6:.2f}M, "
              f"峰值 lr: {stage_lr:.3e}, 静态分支冻结: {model.static_frozen}")

        no_improve_count = 0

        for epoch_in_stage in range(first_epoch, stage_epochs + 1):
            global_epoch += 1
            epoch_start = time.time()

            # ---------- 每个 epoch 重建 train_loader ----------
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                prefetch_factor=2 if NUM_WORKERS > 0 else None,
                multiprocessing_context=_loader_context(NUM_WORKERS),
                worker_init_fn=_worker_init_fn,
            )

            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            desc = (f"[{stage['name']}] {epoch_in_stage}/{stage_epochs}")
            pbar = tqdm(train_loader, desc=desc)

            for batch in pbar:
                inputs, dynamic, labels = _unpack_batch(batch)
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                if dynamic is not None:
                    dynamic = dynamic.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs, dynamic) if dynamic is not None else model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # 梯度裁剪：阶段 1 动态分支随机初始化时防止偶发大梯度
                clip = config.get('grad_clip', 5.0)
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), clip)

                optimizer.step()
                scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (pred == labels).sum().item()

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct_train/max(total_train,1):.4f}",
                    'lr': f"{optimizer.param_groups[-1]['lr']:.2e}",
                })

            del train_loader
            gc.collect()

            epoch_loss = running_loss / max(total_train, 1)
            epoch_acc = correct_train / max(total_train, 1)
            print(f"Epoch {global_epoch} (阶段{stage_idx+1} 第{epoch_in_stage}轮) "
                  f"Train | Loss: {epoch_loss:.4f} | Char Acc: {epoch_acc:.4f}")

            # ---------- 评估前先存一次，防止评估崩溃丢进度 ----------
            _save(latest_path, stage_idx, epoch_in_stage, optimizer, scheduler)

            test_page_acc = None
            if epoch_in_stage % EVAL_FREQUENCY == 0:
                test_page_acc = evaluate_page_level(
                    model, metadata_file, global_label_map, DEVICE,
                    num_workers=min(8, NUM_WORKERS),
                    page_workers=config.get('page_workers', 8),
                    segment_workers=config.get('segment_workers', 8),
                    apply_drop=False,
                    return_dynamic=return_dynamic,
                    seq_len=SEQ_LEN,
                    dynamic_speed_clip=DYNAMIC_SPEED_CLIP,
                )
                print(f"Epoch {global_epoch} Test | 页面级准确率: {test_page_acc:.4f}")

                if test_page_acc > best_page_acc:
                    best_page_acc = test_page_acc
                    no_improve_count = 0
                    _save(best_path, stage_idx, epoch_in_stage, optimizer, scheduler)
                    print(f"🎉 新最佳模型 (页面 Acc: {best_page_acc:.4f}) → {best_path}")
                else:
                    no_improve_count += 1
                    print(f"⚠️ 验证未提升 ({no_improve_count}/{PATIENCE})")

                _save(latest_path, stage_idx, epoch_in_stage, optimizer, scheduler)

            epoch_time = time.time() - epoch_start
            _append_log(log_path, [
                stage_idx + 1, stage['name'], epoch_in_stage, global_epoch,
                f"{epoch_loss:.4f}", f"{epoch_acc:.4f}",
                f"{test_page_acc:.4f}" if test_page_acc is not None else "",
                f"{optimizer.param_groups[-1]['lr']:.8f}",
                f"{epoch_time:.1f}",
            ])

            # 早停只在当前阶段内生效，不会跨阶段触发
            if no_improve_count >= PATIENCE:
                print(f"[EARLY STOP] 阶段 {stage['name']} 连续 {PATIENCE} 次未提升，"
                      f"提前结束该阶段。")
                break

            print("-" * 50)

        # ---------- 阶段结束保存 ----------
        stage_path = os.path.join(ckpt_dir, f'{prefix}_{stage["name"]}.pth')
        _save(stage_path, stage_idx, stage_epochs)
        print(f"💾 阶段 {stage['name']} 结束，权重已保存至 {stage_path}")

        if stage_idx == start_stage_idx:
            start_epoch_in_stage = 1  # 后续阶段从第 1 轮开始

    print("\n" + "=" * 74)
    print(f"🏁 训练完成 | 模式 = {mode} | 最佳页面级准确率 = {best_page_acc*100:.2f}%")
    print(f"   最佳权重: {best_path}")
    print(f"   训练日志: {log_path}")
    print("=" * 74)

    return {'best_page_acc': best_page_acc, 'best_path': best_path,
            'log_path': log_path}
