"""
双路融合训练入口（消融实验 A/B/C 共用）。

用法:
    # 实验 C：双路融合
    python scripts/train_dual.py --mode dual

    # 实验 A：纯静态基线
    python scripts/train_dual.py --mode static

    # 实验 B：纯动态分支
    python scripts/train_dual.py --mode dynamic

    # 覆盖若干超参（便于扫参）
    python scripts/train_dual.py --mode dual --set seq_len=300 d_model=96

    # 从零开始（忽略已有断点）
    python scripts/train_dual.py --mode dual --no_resume

说明:
    - 三种模式共用同一份 config，保证数据管道 / 增强 / 评估协议完全一致，
      消融对比才具备可比性。
    - 每个模式有独立的 checkpoint 前缀与日志文件，互不覆盖。
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from src.training.trainer_dual import train_dual, resolve_stages  # noqa: E402
from src.data.dataset import STATIC_FEATURE_VERSION  # noqa: E402


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def flatten_config(config):
    """
    把 model / data / training 三节并入顶层，同时保留 checkpoint / logging
    两节为嵌套字典（trainer_dual 会以嵌套方式读取它们）。
    """
    flat = {}
    for key in ('model', 'data', 'training'):
        flat.update(config.get(key, {}) or {})
    flat['checkpoint'] = config.get('checkpoint', {}) or {}
    flat['logging'] = config.get('logging', {}) or {}
    flat['metadata_file'] = config.get('metadata_file', 'data/features/metadata.csv')
    flat['seed'] = config.get('seed', 42)
    return flat


def apply_overrides(flat, overrides):
    """--set key=value 的简易覆盖，值用 yaml 解析以自动识别类型。"""
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f"--set 参数格式应为 key=value，收到: {item}")
        key, _, raw = item.partition('=')
        key = key.strip()
        value = yaml.safe_load(raw.strip())
        flat[key] = value
        print(f"   ⚙️  覆盖 {key} = {value!r}")
    return flat


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dual_stream.yaml')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['static', 'dynamic', 'dual'],
                        help='消融实验模式，覆盖 config 中的 mode')
    parser.add_argument('--prefix', type=str, default=None,
                        help='checkpoint 前缀，默认 <mode>_stream')
    parser.add_argument('--set', dest='overrides', nargs='*', default=None,
                        help='覆盖任意配置项，如 seq_len=300 d_model=96')
    parser.add_argument('--no_resume', action='store_true',
                        help='忽略已有断点，从阶段 1 第 1 轮开始')
    parser.add_argument('--static_init', type=str, default=None,
                        help='覆盖 model.static_init_path；传 none 表示不热启动静态分支')
    parser.add_argument('--static_epochs', type=int, default=None,
                        help='覆盖 static 模式的阶段轮数（热启动微调时用更短的余弦周期）')
    parser.add_argument('--static_lr', type=float, default=None,
                        help='覆盖 static 模式的阶段学习率（占位值，会被 lr_base×batch/256 缩放）')
    args = parser.parse_args()

    raw = load_config(args.config)
    flat = flatten_config(raw)

    # 命令行 mode 优先级最高
    if args.mode:
        flat['mode'] = args.mode
    mode = flat.get('mode', 'dual')
    flat['mode'] = mode

    # 每个模式独立的输出路径，避免互相覆盖
    flat.setdefault('checkpoint', {})
    flat.setdefault('logging', {})
    if args.prefix:
        flat['checkpoint']['prefix'] = args.prefix
    else:
        flat['checkpoint'].setdefault('prefix', f'{mode}_stream')
    prefix = flat['checkpoint']['prefix']
    flat['logging'].setdefault('log_file', f'{prefix}_training_log.csv')
    if args.prefix:
        flat['logging']['log_file'] = f'{prefix}_training_log.csv'

    flat = apply_overrides(flat, args.overrides)

    # --static_init 覆盖（'none' / 'null' / '' 表示不热启动）
    if args.static_init is not None:
        val = args.static_init.strip()
        flat['static_init_path'] = None if val.lower() in ('none', 'null', '') else val
        print(f"   ⚙️  static_init_path = {flat['static_init_path']!r}")

    # --static_epochs / --static_lr：只改 static 模式的阶段配置
    if args.static_epochs is not None or args.static_lr is not None:
        by_mode = flat.setdefault('stages_by_mode', {})
        static_stages = by_mode.setdefault('static', [
            {'name': 'static_baseline', 'epochs': 50, 'lr': 1.0,
             'freeze_static': False, 'static_lr_scale': 1.0},
        ])
        if args.static_epochs is not None:
            for st in static_stages:
                st['epochs'] = args.static_epochs
            print(f"   ⚙️  static 阶段轮数 = {args.static_epochs}")
        if args.static_lr is not None:
            for st in static_stages:
                st['lr'] = args.static_lr
            print(f"   ⚙️  static 阶段 lr(占位) = {args.static_lr}")

    # 基线/微调实验的起点提示
    if mode == 'static':
        if flat.get('static_init_path'):
            print(f"   ℹ️  纯静态将从 {flat['static_init_path']} 热启动"
                  f"（换成 --static_init none 可从零训练）")
        else:
            print("   ℹ️  纯静态从零训练（骨干网预训练权重 + 随机分类头）")

    if args.no_resume:
        flat['resume'] = False

    set_seed(flat.get('seed', 42))

    # ========== 打印配置确认 ==========
    print("=" * 74)
    print(f"配置加载确认 | mode = {mode} | config = {args.config}")
    print(f"  batch_size        = {flat.get('batch_size')}")
    print(f"  num_workers       = {flat.get('num_workers')}")
    print(f"  seq_len           = {flat.get('seq_len')}")
    print(f"  dynamic_speed_clip= {flat.get('dynamic_speed_clip')}")
    print(f"  d_model / nhead   = {flat.get('d_model')} / {flat.get('nhead')}")
    print(f"  num_layers        = {flat.get('num_layers')}")
    print(f"  dynamic_dim       = {flat.get('dynamic_dim')}")
    print(f"  pooling           = {flat.get('pooling')}")
    print(f"  feature_dim       = {flat.get('feature_dim')}")
    print(f"  drop_path         = {flat.get('drop_path')}")
    print(f"  dropout_head      = {flat.get('dropout_head')}")
    print(f"  fusion_dropout    = {flat.get('fusion_dropout')}")
    print(f"  static_init_path  = {flat.get('static_init_path')}")
    print(f"  pretrained_path   = {flat.get('pretrained_path')}")
    print(f"  静态特征          = 1 位图 + 63 签名（{STATIC_FEATURE_VERSION}，按原文实现）")
    print(f"  checkpoint prefix = {prefix}")
    print(f"  resume            = {flat.get('resume')}")
    print("-" * 74)
    print("训练阶段:")
    base_scale = (flat.get('lr_base', 0.000667) * flat.get('batch_size', 192) / 256
                  if mode == 'static' else 1.0)
    for i, st in enumerate(resolve_stages(flat)):
        print(f"  {i+1}. {st['name']:<24} epochs={st['epochs']:<3} "
              f"lr={float(st['lr']) * base_scale:.3e} "
              f"freeze_static={st.get('freeze_static')} "
              f"static_lr_scale={st.get('static_lr_scale')}")
    print("=" * 74)

    train_dual(flat)


if __name__ == '__main__':
    main()
