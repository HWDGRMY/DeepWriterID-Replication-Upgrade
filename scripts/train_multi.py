"""
统一训练入口 —— 单独训练一个模型，或一起训练多个模型。

一份内存数据、一个 DataLoader，同时喂 runs 里的所有模型（CPU 特征提取只做一次）。

用法:
    # 1) 一起训练（配置里默认的两个模型：ConvNeXt + DCNN）
    python scripts/train_multi.py

    # 2) 只单独训练其中一个（从同一份配置里挑）
    python scripts/train_multi.py --only dcnn
    python scripts/train_multi.py --only convnext

    # 3) 只训练指定的几个
    python scripts/train_multi.py --runs convnext dcnn

    # 4) 列出配置里有哪些模型
    python scripts/train_multi.py --list

    # 5) 覆盖配置
    python scripts/train_multi.py --set epochs=30 batch_size=128
    python scripts/train_multi.py --run_override dcnn lr=0.0003
    python scripts/train_multi.py --no_resume

产物（--only / --runs 会自动给文件名加后缀，不会和「一起训练」的产物冲突）:
    outputs/checkpoints/<group>_latest.pth     组合断点（含优化器状态，可续训）
    outputs/checkpoints/<prefix>_best.pth      每个模型的最佳权重（标准格式）
    outputs/logs/<log_file>                    逐轮指标，每个模型 5 列
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

from src.training.trainer_multi import train_multi  # noqa: E402
from src.data.dataset import STATIC_FEATURE_VERSION  # noqa: E402


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def flatten(config):
    """把 model / data / training 展平，同时保留 checkpoint / logging 嵌套。"""
    flat = {}
    for key in ('model', 'data', 'training'):
        flat.update(config.get(key, {}) or {})
    flat['checkpoint'] = dict(config.get('checkpoint', {}) or {})
    flat['logging'] = dict(config.get('logging', {}) or {})
    # 嵌套字典必须整块带过来：漏掉的话配置文件里的 eval_20test.n_test 会被
    # 静默忽略，训练器只能吃代码默认值（恰好相同就更难发现）。
    flat['eval_20test'] = dict(config.get('eval_20test', {}) or {})
    flat['runs'] = [dict(r) for r in (config.get('runs') or [])]
    flat['metadata_file'] = config.get('metadata_file', 'data/features/metadata.csv')
    flat['seed'] = config.get('seed', 42)
    return flat


def select_runs(flat, only=None, names=None):
    """
    从配置里挑选要训练的模型。

    · 都不指定   → 全部（一起训练）
    · --only X   → 只训练 X（单独训练）
    · --runs ... → 只训练列出的几个

    挑了子集时自动给组合断点与日志加后缀，避免与「全部一起训练」的产物互相覆盖。
    """
    all_runs = flat['runs']
    available = [r.get('name') for r in all_runs]

    if only and names:
        raise SystemExit('--only 与 --runs 只能用一个')
    want = [only] if only else (list(names) if names else None)
    if want is None:
        return flat, available, False

    missing = [w for w in want if w not in available]
    if missing:
        raise SystemExit(f"配置里没有这些模型: {missing}\n可选: {available}")

    flat['runs'] = [r for r in all_runs if r.get('name') in want]
    order = {n: i for i, n in enumerate(want)}
    flat['runs'].sort(key=lambda r: order[r['name']])

    tag = '-'.join(want)
    ck = flat['checkpoint']
    ck['group'] = f"{ck.get('group', 'multi_arch')}_{tag}"
    log = flat['logging']
    if log.get('log_file'):
        stem, dot, ext = log['log_file'].rpartition('.')
        log['log_file'] = f'{stem}_{tag}.{ext}' if dot else f"{log['log_file']}_{tag}"
    return flat, [r['name'] for r in flat['runs']], True


def apply_overrides(flat, overrides):
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f"--set 格式应为 key=value，收到: {item}")
        key, _, raw = item.partition('=')
        key = key.strip()
        value = yaml.safe_load(raw.strip())
        if '.' in key:
            head, _, tail = key.partition('.')
            flat.setdefault(head, {})[tail] = value
        else:
            flat[key] = value
        print(f"   ⚙️  覆盖 {key} = {value!r}")
    return flat


def apply_run_overrides(flat, run_name, overrides):
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f"--run_override 格式应为 key=value，收到: {item}")
        key, _, raw = item.partition('=')
        value = yaml.safe_load(raw.strip())
        hit = False
        for r in flat['runs']:
            if r.get('name') == run_name:
                r[key.strip()] = value
                hit = True
        if not hit:
            raise ValueError(f"--run_override: 当前待训练列表里没有 {run_name!r}")
        print(f"   ⚙️  [{run_name}] 覆盖 {key.strip()} = {value!r}")
    return flat


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/multi_arch.yaml')
    ap.add_argument('--only', type=str, default=None,
                    help='只单独训练这一个模型（等价于 --runs <名字>）')
    ap.add_argument('--runs', nargs='+', default=None,
                    help='只训练列出的这几个模型')
    ap.add_argument('--list', action='store_true',
                    help='列出配置里有哪些模型，然后退出')
    ap.add_argument('--set', dest='overrides', nargs='*', action='append',
                    default=None,
                    help='覆盖顶层配置项，如 --set epochs=30 batch_size=128。'
                         '可以写多次，累积生效（不会互相覆盖）。')
    ap.add_argument('--run_override', nargs='+', default=None,
                    metavar=('RUN_NAME', 'KEY=VALUE'),
                    help='覆盖某个模型的字段，如 --run_override dcnn lr=0.0003')
    ap.add_argument('--no_resume', action='store_true')
    args = ap.parse_args()

    flat = flatten(load_config(args.config))
    total_runs = len(flat['runs'])

    if args.list:
        print(f"配置 {args.config} 里有 {total_runs} 个模型:")
        for r in flat['runs']:
            print(f"  - {r.get('name'):14s} arch={r.get('arch'):9s} "
                  f"mode={r.get('mode'):8s} lr={r.get('lr')} "
                  f"prefix={r.get('prefix')}")
        print("\n单独训练其中一个:  python scripts/train_multi.py --only <名字>")
        print("一起训练全部:      python scripts/train_multi.py")
        return

    flat, selected, is_subset = select_runs(flat, args.only, args.runs)
    # --set 用 action='append'，多次出现会累积；这里摊平成一个列表
    all_overrides = [x for group in (args.overrides or []) for x in group]
    flat = apply_overrides(flat, all_overrides)

    if args.run_override:
        if len(args.run_override) < 2:
            raise SystemExit('--run_override 需要至少两个参数: <run名> key=value')
        flat = apply_run_overrides(flat, args.run_override[0], args.run_override[1:])

    if args.no_resume:
        flat['resume'] = False

    set_seed(flat.get('seed', 42))

    print("=" * 80)
    print(f"配置: {args.config}")
    mode_txt = "单独训练" if len(flat['runs']) == 1 else "一起训练（共享管道）"
    print(f"模式: {mode_txt} | 模型: {selected}")
    if is_subset:
        print(f"      （从配置里的 {total_runs} 个中挑了 {len(flat['runs'])} 个，"
              f"产物已加后缀避免覆盖）")
    print(f"  batch_size={flat.get('batch_size')}  epochs={flat.get('epochs')}  "
          f"num_workers={flat.get('num_workers')}")
    print(f"  page_workers={flat.get('page_workers')}  "
          f"segment_workers={flat.get('segment_workers')}")
    print(f"  metadata_file={flat.get('metadata_file')}")
    print(f"  静态特征: 1 位图 + 63 签名（{STATIC_FEATURE_VERSION}，按原文实现）")
    print(f"  组合断点: {flat['checkpoint'].get('group')}")
    print(f"  日志: {flat['logging'].get('log_file')}")
    for r in flat['runs']:
        print(f"    - {r.get('name'):14s} arch={r.get('arch'):9s} "
              f"mode={r.get('mode'):8s} lr={r.get('lr')} "
              f"opt={r.get('optimizer')} prefix={r.get('prefix')}")
    print("=" * 80)

    train_multi(flat)


if __name__ == '__main__':
    main()
