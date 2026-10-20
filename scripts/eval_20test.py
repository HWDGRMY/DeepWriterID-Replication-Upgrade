"""
20-test 增强评估（独立于训练，可随时对已保存的检查点重跑）。

什么是 20-test
--------------
论文口径的稳健性评估：对测试集每轮**重新随机** DropSegment（随机丢弃若干笔画）
后推一次，跑 N 轮（通常 20），把每轮的页面级概率累加平均，得到最终准确率。
它衡量的是「模型对切分/丢笔这种扰动的稳定性」，比单次确定性评估更严格。

为什么必须共享增强
------------------
如果模型 A 和模型 B 各自独立采样 20 次，它们吃的增强样本不同，
两组的差异里就混进了「增强随机性」这个变量，**这样算出的配对 t 检验无效**。
本脚本每轮只建一个 DataLoader，所有模型前向同一批张量，所以：
  · 模型之间严格配对，可直接做配对检验
  · 数据管道开销只付一次

用法
----
    # 自动发现 outputs/checkpoints 下所有 *_best.pth
    python scripts/eval_20test.py

    # 指定检查点
    python scripts/eval_20test.py --ckpt outputs/checkpoints/arch_cn_best.pth \
                                       outputs/checkpoints/arch_dcnn_best.pth

    # 改轮数 / 批量（显存不够就降 batch）
    python scripts/eval_20test.py --n_test 20 --batch_size 128 --num_workers 8

产物
----
    outputs/logs/<group>_20test_records.csv   逐轮：每轮的独立准确率 + 累积准确率
    outputs/logs/20test_summary.csv           汇总：final / mean ± std（论文直接引用）

重要
----
特征编码版本（`static_feature_version`）**从每个检查点自己的 config 里读取**，
不是用命令行或全局配置的默认值。版本不同意味着 64 通道的语义可能已经改变，
用错版本的权重评估会让准确率莫名其妙地偏低，而且表面一切正常。

如果选中的多个检查点版本不一致（或缺少版本记录），脚本会直接报错退出。
"""

from __future__ import annotations

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from src.models.dual_stream import build_model                      # noqa: E402
from src.evaluation.evaluator import evaluate_page_level_20test_multi  # noqa: E402
from src.data.dataset import STATIC_FEATURE_VERSION                 # noqa: E402


# ----------------------------------------------------------------------
# 从检查点重建模型
# ----------------------------------------------------------------------

def _infer_num_classes(state_dict) -> int:
    """从权重形状里猜类别数（找不到就返回 None）。"""
    for k, v in state_dict.items():
        if k.endswith('classifier.1.weight') or k.endswith('output.1.weight'):
            return int(v.shape[0])
    return None


def load_ckpt_model(path, num_classes, device, verbose=True):
    """
    读检查点 → 重建模型 → 加载权重。

    返回 (model, meta)，meta 含训练时用的 arch / mode / 流水线变体。
    架构与流水线**一律取自检查点**，不取脚本默认值。
    """
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = dict(ck.get('config') or {})
    state = ck.get('model_state_dict', ck)

    # 老检查点可能没有 config，只能从权重键名反推
    arch = cfg.get('arch') or ck.get('arch')
    if not arch:
        arch = 'dcnn' if any(k.startswith('conv1.0.weight') for k in state) \
            else 'convnext'
    mode = cfg.get('mode') or ck.get('mode') or 'static'

    n_ck = _infer_num_classes(state)
    if n_ck is not None and n_ck != num_classes:
        raise RuntimeError(
            f"类别数不匹配：{os.path.basename(path)} 是 {n_ck} 类，"
            f"当前 metadata 解析出 {num_classes} 类。\n"
            f"评估必须用与训练**同一份 metadata 划分**，否则标签索引对不上。")

    build_cfg = dict(cfg)
    build_cfg.update({
        'arch': arch,
        'mode': mode,
        # 推理阶段关掉所有随机正则
        'drop_path': 0.0,
        'dropout_head': 0.0,
        'dropout_classifier': 0.0,
        'dcnn_dropout': 0.0,
        'dcnn_dropout_classifier': 0.0,
        'dynamic_dropout': 0.0,
        'fusion_dropout': 0.0,
        'pretrained': False,
    })
    model = build_model(build_cfg, num_classes, pretrained_path=None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"   ⚠️ 键不匹配 missing={len(missing)} unexpected={len(unexpected)}")
        for k in missing[:3]:
            print(f"      missing: {k}")
        for k in unexpected[:3]:
            print(f"      unexpected: {k}")
    model = model.to(device).eval()

    meta = {
        'path': path,
        'name': cfg.get('run_name') or os.path.basename(path).replace('_best.pth', ''),
        'arch': arch,
        'mode': mode,
        'epoch': ck.get('epoch'),
        'best_acc': ck.get('best_acc'),
        'feature_version': cfg.get('static_feature_version'),
        'needs_dynamic': mode in ('dynamic', 'dual'),
    }
    if verbose:
        print(f"✅ {meta['name']:14s} arch={arch:9s} mode={mode:8s} "
              f"epoch={meta['epoch']} train_acc={meta['best_acc']}")
        print(f"   特征版本: {meta['feature_version']}")
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='*', default=None,
                    help='检查点路径；不给则自动搜索 ckpt_dir 下的 *_best.pth')
    ap.add_argument('--ckpt_dir', type=str, default='outputs/checkpoints')
    ap.add_argument('--metadata', type=str, default='data/features/metadata.csv')
    ap.add_argument('--n_test', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--page_workers', type=int, default=4)
    ap.add_argument('--segment_workers', type=int, default=4)
    ap.add_argument('--group', type=str, default='20test')
    ap.add_argument('--out_dir', type=str, default='outputs/logs')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备 = {device}")
    if device.type == 'cuda':
        print(f"GPU = {torch.cuda.get_device_name(0)}")

    # ---- metadata → 标签映射（与训练一致）----
    df = pd.read_csv(args.metadata, encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)
    all_writer_ids = sorted(df['writer_id'].unique())
    global_label_map = {wid: i for i, wid in enumerate(all_writer_ids)}
    num_classes = len(global_label_map)
    n_test_pages = int((df['split'] == 'Test').sum())
    print(f"📊 类别数 = {num_classes}，Test 页面数 = {n_test_pages}")

    # ---- 找检查点 ----
    paths = args.ckpt
    if not paths:
        paths = sorted(glob.glob(os.path.join(args.ckpt_dir, '*_best.pth')))
    if not paths:
        raise SystemExit(f"在 {args.ckpt_dir} 下没找到 *_best.pth，请用 --ckpt 指定")
    print(f"\n找到 {len(paths)} 个检查点:")
    for p in paths:
        print(f"   {p}")

    # ---- 逐个加载 ----
    print()
    models, metas = {}, []
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"检查点不存在: {p}")
        m, meta = load_ckpt_model(p, num_classes, device)
        models[meta['name']] = m
        metas.append(meta)

    # ---- 特征编码版本必须一致，否则权重与特征语义对不上 ----
    versions = {m['feature_version'] for m in metas}
    if len(versions) > 1 or None in versions:
        print("\n" + "!" * 78)
        print("❌ 选中的检查点特征编码版本不一致（或缺少版本记录）：")
        for m in metas:
            print(f"   {m['name']:14s} feature_version={m['feature_version']}")
        print(f"\n   当前代码版本 = {STATIC_FEATURE_VERSION}")
        print("   版本不同意味着 64 通道的语义可能已经改变，权重不可混用、")
        print("   也不能在同一次 20-test 里配对比较。请分开跑：")
        print("     python scripts/eval_20test.py --ckpt <版本相同的那些>")
        print("!" * 78)
        raise SystemExit(2)

    print(f"\n✅ 特征版本一致: {versions.pop()}")

    # ---- 跑 20-test（共享增强）----
    out_records = os.path.join(args.out_dir, f'{args.group}_records.csv')
    res = evaluate_page_level_20test_multi(
        models, args.metadata, global_label_map, device,
        n_test=args.n_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        page_workers=args.page_workers,
        segment_workers=args.segment_workers,
        apply_drop=True,
        needs_dynamic={n: m['needs_dynamic'] for n, m in zip(models, metas)},
        seq_len=200,
        dynamic_speed_clip=10.0,
        log_path=out_records,
    )

    # ---- 汇总 ----
    print("\n" + "=" * 78)
    print(f"{args.n_test}-test 结果汇总（所有模型共享同一批增强，配对有效）")
    print("=" * 78)
    print(f"{'模型':<16}{'架构':<12}{'最终累积':>10}{'单轮均值':>12}{'标准差':>10}")
    print("-" * 78)
    for m in metas:
        r = res['models'][m['name']]
        print(f"{m['name']:<16}{m['arch']:<12}"
              f"{r['final_acc']*100:>9.2f}%{r['mean_acc']*100:>11.2f}%"
              f"{r['std_acc']*100:>9.2f}%")
    print("=" * 78)
    print(f"逐轮记录: {out_records}")
    print(f"汇总    : {os.path.join(args.out_dir, '20test_summary.csv')}")


if __name__ == '__main__':
    main()
