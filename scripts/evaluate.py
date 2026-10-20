import sys
import os
import yaml
import torch
import pandas as pd

# 获取项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.models.backbone import ConvNeXtBackbone          # 改动1: 换模型
from src.evaluation.evaluator import evaluate_page_level

if __name__ == '__main__':
    config_path = os.path.join(BASE_DIR, 'configs', 'convnext.yaml')
    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    metadata_file = os.path.join(BASE_DIR, 'data', 'features', 'metadata.csv')
    checkpoint_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'convnext_best.pth')   # 改动2: 权重文件名
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 构建全局映射
    df = pd.read_csv(metadata_file, encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)
    all_writer_ids = sorted(df['writer_id'].unique())
    global_label_map = {wid: idx for idx, wid in enumerate(all_writer_ids)}

    # 2. 加载模型
    model = ConvNeXtBackbone(                              # 改动3: 换实例化
        num_classes=len(global_label_map),
        in_chans=64,
        feature_dim=512,
    ).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)   # 改动4: strict=True
        else:
            model.load_state_dict(checkpoint, strict=True)
        print(f"✅ 成功加载模型: {checkpoint_path}")
    else:
        print(f"⚠️ 未找到模型文件: {checkpoint_path}，将使用未训练的随机权重")

    # 3. 开始评估
    evaluate_page_level(model, metadata_file, global_label_map, device)