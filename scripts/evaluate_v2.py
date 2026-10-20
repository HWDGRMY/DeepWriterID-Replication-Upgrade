"""
用 v2 的 20-test evaluator 测试 v2 的 ConvNeXt V2 模型。

用法:
    python scripts/evaluate_v2.py --ckpt outputs/checkpoints/convnext_best.pth --n_test 20
"""

import os
import sys
import argparse
import torch
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.backbone import ConvNeXtBackbone
from src.evaluation.evaluator import evaluate_page_level_20test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='outputs/checkpoints/convnext_best.pth')
    parser.add_argument('--config', type=str, default='configs/convnext.yaml')
    parser.add_argument('--metadata', type=str,
                        default='data/features/metadata.csv')
    parser.add_argument('--n_test', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_name', type=str, default='20test_records_v2.csv',
                        help='输出 CSV 文件名')
    args = parser.parse_args()

    # ========== 加载 config ==========
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 展平
    flat = {}
    for key in ['model', 'data', 'training', 'checkpoint', 'logging']:
        flat.update(config.get(key, {}))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 设备: {device}")

    # ========== 构造全局 label map ==========
    df = pd.read_csv(args.metadata, encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)
    all_writer_ids = sorted(df['writer_id'].unique())
    global_label_map = {wid: idx for idx, wid in enumerate(all_writer_ids)}
    num_classes = len(global_label_map)
    print(f"📊 类别数: {num_classes}")

    # ========== 加载 v2 ConvNeXt V2 ==========
    model = ConvNeXtBackbone(
        num_classes=num_classes,
        in_chans=64,
        feature_dim=512,
        drop_path_rate=flat.get('drop_path', 0.1),
        dropout_head=flat.get('dropout_head', 0.3),
        dropout_classifier=flat.get('dropout_classifier', 0.4),
        pretrained=False,
    ).to(device)

    if not os.path.exists(args.ckpt):
        print(f"❌ 找不到权重文件: {args.ckpt}")
        return

    ckpt = torch.load(args.ckpt, map_location=device)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ 已加载 v2 ConvNeXt V2 权重: {args.ckpt}")
        if 'epoch' in ckpt:
            print(f"   来源 epoch: {ckpt['epoch']}, best_acc: {ckpt.get('best_acc', '?')}")
    else:
        model.load_state_dict(ckpt)
        print(f"✅ 已加载 v2 ConvNeXt V2 权重（裸权重）: {args.ckpt}")

    # ========== 20-test 评估 ==========
    acc = evaluate_page_level_20test(
        model, args.metadata, global_label_map, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_test=args.n_test,
        log_name=args.log_name,
    )

    print(f"\n{'='*60}")
    print(f"最终 v2 ConvNeXt V2（{args.n_test} test）: {acc*100:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()