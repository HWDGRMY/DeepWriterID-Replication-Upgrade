"""
ConvNeXt V2 训练器：每个 epoch 重建 train_loader（避免 persistent worker 与测试进程冲突）。

关键设计：
- 训练阶段：每个 epoch 新建 train_loader（16 worker），epoch 结束自动销毁
- 测试阶段：训练 worker 已销毁，可以安全用 8 worker 跑测试
- 保存时机：训练结束后、评估前，防止评估崩溃丢失全部
- 预训练：支持从本地 .safetensors 加载
- 调度器：Warmup + 余弦退火，iteration 级 step
- 断点续训：完整保存/恢复模型、优化器、调度器状态
"""

import os
import sys
import csv
import time
import gc
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import pandas as pd
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset import WPTTDataset
from src.models.backbone import ConvNeXtBackbone
from src.evaluation.evaluator import evaluate_page_level

import random as _random
import numpy as _np

def _worker_init_fn(worker_id):
    """每个 DataLoader worker 独立随机种子，保证 DropSegment 真随机。"""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    _random.seed(worker_seed)
    _np.random.seed(worker_seed)



def _pad_to_96(tensor):
    """v1.0 继承：54×54 → 96×96。"""
    pad = (21, 21, 21, 21)
    return torch.nn.functional.pad(tensor, pad, mode='constant', value=0)


def _init_log_file(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_char_acc',
                'test_page_acc', 'lr', 'epoch_time_sec'
            ])
        print(f"📝 已创建训练日志: {log_path}")


def _append_log(log_path, row):
    with open(log_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train(config):
    # ========== 读取超参数 ==========
    BATCH_SIZE = config.get('batch_size', 192)
    EPOCHS = config.get('epochs', 50)
    LR_BASE = config.get('lr_base', 0.000667)
    INITIAL_LR = LR_BASE * BATCH_SIZE / 256
    WEIGHT_DECAY = config.get('weight_decay', 0.05)
    WARMUP_EPOCHS = config.get('warmup_epochs', 1)
    EVAL_FREQUENCY = config.get('eval_frequency', 1)
    PATIENCE = config.get('patience', 10)
    DROP_PATH = config.get('drop_path', 0.1)
    DROPOUT_HEAD = config.get('dropout_head', 0.3)
    DROPOUT_CLASSIFIER = config.get('dropout_classifier', 0.4)
    PRETRAINED = config.get('pretrained', False)
    PRETRAINED_PATH = config.get('pretrained_path', None)
    NUM_WORKERS = config.get('num_workers', 16)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(f"🔧 配置: batch={BATCH_SIZE}, lr={INITIAL_LR:.6f}, "
          f"pretrained={PRETRAINED}, warmup={WARMUP_EPOCHS}, "
          f"eval_freq={EVAL_FREQUENCY}, workers={NUM_WORKERS}, "
          f"drop_path={DROP_PATH}, dropout_head={DROPOUT_HEAD}, "
          f"dropout_classifier={DROPOUT_CLASSIFIER}")

    # ========== 数据路径 ==========
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metadata_file = os.path.join(BASE_DIR, 'data', 'features', 'metadata.csv')

    print("正在加载数据集 ...")
    full_df = pd.read_csv(metadata_file, encoding='utf-8-sig')
    full_df['writer_id'] = full_df['writer_id'].astype(str)
    all_writer_ids = sorted(full_df['writer_id'].unique())
    global_label_map = {wid: idx for idx, wid in enumerate(all_writer_ids)}
    num_classes = len(global_label_map)

    # ========== 数据增强与 Dataset（只创建一次） ==========
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        _pad_to_96
    ])

    train_dataset = WPTTDataset(metadata_file, split='Train', transform=train_transform)
    print(f"📊 训练集样本数: {len(train_dataset)}")
    print(f"📊 作者（类别）总数: {num_classes}")

    # ========== 模型 ==========
    model = ConvNeXtBackbone(
        num_classes=num_classes,
        in_chans=64,
        feature_dim=512,
        drop_path_rate=DROP_PATH,
        dropout_head=DROPOUT_HEAD,
        dropout_classifier=DROPOUT_CLASSIFIER,
        pretrained=PRETRAINED,
        pretrained_path=PRETRAINED_PATH,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数量: {total_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()

    # ========== 优化器与调度器 ==========
    optimizer = optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    # 因为 train_loader 在循环里创建，这里只能估算 steps_per_epoch
    steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
    warmup_steps = max(WARMUP_EPOCHS * steps_per_epoch, 1)
    cosine_steps = max((EPOCHS - WARMUP_EPOCHS) * steps_per_epoch, 1)

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # ========== 路径与状态 ==========
    log_path = os.path.join(BASE_DIR, 'outputs', 'logs', 'convnext_training_log.csv')
    latest_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'convnext_latest.pth')
    best_model_path = os.path.join(BASE_DIR, 'outputs', 'checkpoints', 'convnext_best.pth')
    os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
    _init_log_file(log_path)

    start_epoch = 1
    best_page_acc = 0.0
    no_improve_count = 0


    # ========== Warm start（仅当没有 latest checkpoint 时） ==========
    WARM_START_PATH = config.get('warm_start_path', None)
    if WARM_START_PATH and os.path.exists(WARM_START_PATH) and not os.path.exists(latest_model_path):
        print(f"🔥 Warm start from {WARM_START_PATH}")
        ws_ckpt = torch.load(WARM_START_PATH, map_location=DEVICE)
        if isinstance(ws_ckpt, dict) and 'model_state_dict' in ws_ckpt:
            model.load_state_dict(ws_ckpt['model_state_dict'])
        else:
            model.load_state_dict(ws_ckpt)
        print(f"✅ Warm start 成功（optimizer/scheduler 将重新初始化）")

    # ========== 断点续训 ==========

    if os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_page_acc = checkpoint.get('best_acc', 0.0)
        if 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✅ 完整断点恢复: Epoch {start_epoch}, Best Page Acc: {best_page_acc:.4f}")
        else:
            print(f"✅ 仅模型权重恢复（optimizer/scheduler 重新初始化）: Epoch {start_epoch}, Best Page Acc: {best_page_acc:.4f}")
    else:
        print("🆕 未找到检查点，将从 Epoch 1 开始全新训练。")

    # ========== 训练主循环 ==========
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start_time = time.time()

        # ========== 每个 epoch 新建 train_loader ==========
        # 关键：不用 persistent_workers，epoch 结束后 worker 自动销毁
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=2,
            multiprocessing_context='fork',
            worker_init_fn=_worker_init_fn,
        )

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (pred == labels).sum().item()

            lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_train/total_train:.4f}",
                'lr': f"{lr:.6f}",
            })

        # ========== 训练结束，销毁 loader 释放 worker ==========
        del train_loader
        gc.collect()

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        print(f"Epoch {epoch} Train | Loss: {epoch_loss:.4f} | Char Acc: {epoch_acc:.4f}")

        # ========== 先保存检查点（评估前） ==========
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_page_acc,
        }, latest_model_path)
        print(f"💾 Epoch {epoch} 权重已保存至 {latest_model_path}")

        # ========== 页面级评估（训练 worker 已销毁，可以安全用 8 worker） ==========
        test_page_acc = None
        if epoch % EVAL_FREQUENCY == 0:
            test_page_acc = evaluate_page_level(
                model, metadata_file, global_label_map, DEVICE,
                num_workers=8,          # 测试 DataLoader 用 8
                page_workers=8,         # 测试 Dataset 页面加载用 8
                segment_workers=8,      # 测试 Dataset 切分用 8
            )
            print(f"Epoch {epoch} Test | 页面级准确率: {test_page_acc:.4f}")

            if test_page_acc > best_page_acc:
                best_page_acc = test_page_acc
                no_improve_count = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_page_acc,
                }, best_model_path)
                print(f"🎉 新最佳模型保存 (页面 Acc: {best_page_acc:.4f})")
            else:
                no_improve_count += 1
                print(f"⚠️ 验证未提升 ({no_improve_count}/{PATIENCE})")

            # 评估后再保存一次 latest，同步 best_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_page_acc,
            }, latest_model_path)

        # 写日志
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        _append_log(log_path, [
            epoch,
            f"{epoch_loss:.4f}",
            f"{epoch_acc:.4f}",
            f"{test_page_acc:.4f}" if test_page_acc is not None else "",
            f"{current_lr:.8f}",
            f"{epoch_time:.1f}",
        ])

        # 早停
        if no_improve_count >= PATIENCE:
            print(f"[EARLY STOP] 验证准确率连续 {PATIENCE} 次未提升，停止训练。")
            break

        print("-" * 50)