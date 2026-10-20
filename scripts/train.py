"""
训练入口：读 config，展平嵌套结构后调用 train()。
"""

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.trainer import train


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def flatten_config(config):
    """把嵌套 config 展平成单层字典，适配 trainer.py 的顶层读取方式。"""
    flat = {}
    flat.update(config.get('model', {}))
    flat.update(config.get('data', {}))
    flat.update(config.get('training', {}))
    flat.update(config.get('checkpoint', {}))
    flat.update(config.get('logging', {}))
    flat['seed'] = config.get('seed', 42)
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/convnext.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    flat_config = flatten_config(config)

    # 打印确认所有参数被正确读取
    print("=" * 60)
    print("配置加载确认：")
    print(f"  batch_size = {flat_config.get('batch_size')}")
    print(f"  lr_base = {flat_config.get('lr_base')}")
    print(f"  epochs = {flat_config.get('epochs')}")
    print(f"  warmup_epochs = {flat_config.get('warmup_epochs')}")
    print(f"  weight_decay = {flat_config.get('weight_decay')}")
    print(f"  drop_path = {flat_config.get('drop_path')}")
    print(f"  dropout_head = {flat_config.get('dropout_head')}")
    print(f"  dropout_classifier = {flat_config.get('dropout_classifier')}")
    print(f"  pretrained = {flat_config.get('pretrained')}")
    print("=" * 60)

    train(flat_config)


if __name__ == '__main__':
    main()