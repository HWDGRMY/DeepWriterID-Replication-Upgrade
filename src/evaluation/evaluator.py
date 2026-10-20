"""
页面级投票评估。

- evaluate_page_level：1-test，训练时每轮调用，默认不开 DropSegment
- evaluate_page_level_20test：20-test，训练完成后调用，默认开 DropSegment
- extract_test_features：导出测试集逐样本特征 / 概率 / 标签，供互补性分析使用

双路（静态 + 动态）支持：
- 自动识别 DataLoader 产出的是 (静态, 标签) 还是 (静态, 动态, 标签)
- return_dynamic=True 时把动态序列一并送进模型

消融实验说明：
- 消融模式由模型自身的 `mode` 决定（static / dynamic / dual），
  评估代码不需要区分；三种模式共用同一套数据管道与评估协议。
- 逐样本误差分解与 CCA 需要**确定性测试集**，请用 apply_drop=False。
"""

import os
import gc
import csv
import sys
import time
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from src.data.dataset import WPTTDataset
# DataLoader 统一构造入口：训练 / 评估 / 20-test 共用同一套参数，
# 避免 num_workers、prefetch_factor、worker_init_fn 各处漂移。
from src.data.dataloader_utils import (
    MP_CONTEXT, loader_context as _loader_context, make_loader as _make_loader,
)


def _pad_to_96(tensor):
    pad = (21, 21, 21, 21)
    return torch.nn.functional.pad(tensor, pad, mode='constant', value=0)


# ============================================================
# 通用小工具
# ============================================================

def _unpack_batch(batch):
    """兼容 (静态, 标签) 与 (静态, 动态, 标签) 两种批次结构。"""
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    if len(batch) == 2:
        return batch[0], None, batch[1]
    raise ValueError(f"无法识别的批次结构，元素个数={len(batch)}")


def _forward(model, inputs, dynamic):
    """按是否存在动态输入调用模型。"""
    if dynamic is None:
        return model(inputs)
    return model(inputs, dynamic)


def _compute_page_acc(probs, sample_pages, page_to_true_label):
    page_votes = defaultdict(list)
    for i, page in enumerate(sample_pages):
        page_votes[page].append(probs[i])

    correct = 0
    for page, char_probs_list in page_votes.items():
        page_avg_prob = np.mean(char_probs_list, axis=0)
        pred_label = np.argmax(page_avg_prob)
        if pred_label == page_to_true_label[page]:
            correct += 1
    return correct / len(page_votes)


def _build_test_meta(metadata_file, global_label_map, sample_pages):
    """返回 (page_to_true_label, sample_true_labels)。"""
    df_test = pd.read_csv(metadata_file, encoding='utf-8-sig')
    df_test['writer_id'] = df_test['writer_id'].astype(str)
    test_df = df_test[df_test['split'] == 'Test'].reset_index(drop=True)

    page_to_true_label = {}
    for page in sorted(set(sample_pages)):
        wid = test_df[test_df['page'] == page].iloc[0]['writer_id']
        page_to_true_label[page] = global_label_map[wid]

    sample_true_labels = np.array(
        [page_to_true_label[p] for p in sample_pages], dtype=np.int64
    )
    return page_to_true_label, sample_true_labels


def _make_dataset(metadata_file, apply_drop, return_dynamic,
                  page_workers, segment_workers, seq_len=200,
                  dynamic_speed_clip=10.0, use_bitmap=True):
    return WPTTDataset(
        metadata_file, split='Test', transform=_pad_to_96,
        page_workers=page_workers,
        segment_workers=segment_workers,
        apply_drop=apply_drop,
        return_dynamic=return_dynamic,
        seq_len=seq_len,
        dynamic_speed_clip=dynamic_speed_clip,
        # 评估**必须**关掉坐标域增强：否则每轮测试集内容都不同，
        # 准确率抖动里就混进了增强随机性，无法判断模型是否真的在进步。
        augment_points=False,
        use_bitmap=use_bitmap,
    )


# ============================================================
# 1-test：训练时每轮调用，不开 DropSegment
# ============================================================

def _topk_correct(probs, labels, k):
    """probs: (N, C)；labels: (N,)。返回 top-k 命中数。"""
    if probs.shape[1] < k:
        k = probs.shape[1]
    topk = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    return int((topk == labels[:, None]).any(axis=1).sum())


def evaluate_page_level(model, metadata_file, global_label_map, device,
                        batch_size=128, num_workers=8,
                        page_workers=8, segment_workers=8,
                        apply_drop=False, return_dynamic=False,
                        seq_len=200, dynamic_speed_clip=10.0,
                        return_details=False):
    """
    页面级投票评估。

    Args:
        return_details: False（默认）时只返回页面级 top-1 准确率（向后兼容）。
            True 时返回一个字典，含更完整的指标，便于写进训练日志 CSV：
              page_acc / page_acc_top5 / char_acc / char_acc_top5 /
              n_pages / n_chars / n_pages_correct
    """
    test_dataset = _make_dataset(metadata_file, apply_drop, return_dynamic,
                                 page_workers, segment_workers,
                                 seq_len, dynamic_speed_clip)
    test_loader = _make_loader(test_dataset, batch_size, num_workers)

    was_training = model.training
    model.eval()
    print(f"\n📊 正在并行推理测试集（apply_drop={apply_drop}, "
          f"return_dynamic={return_dynamic}）...")

    df_test = pd.read_csv(metadata_file, encoding='utf-8-sig')
    df_test['writer_id'] = df_test['writer_id'].astype(str)
    test_df = df_test[df_test['split'] == 'Test'].reset_index(drop=True)

    page_names = []
    for page_idx, _, _ in test_dataset.samples:
        page_names.append(test_df.iloc[page_idx]['page'])

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="并行推理"):
            inputs, dynamic, labels = _unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            if dynamic is not None:
                dynamic = dynamic.to(device, non_blocking=True)
            outputs = _forward(model, inputs, dynamic)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # ---- 伪字符级指标 ----
    char_pred = all_probs.argmax(1)
    n_chars = int(len(all_labels))
    char_correct = int((char_pred == all_labels).sum())
    char_correct_top5 = _topk_correct(all_probs, all_labels, 5)

    # ---- 页面级投票 ----
    page_votes = defaultdict(list)
    for prob, page in zip(all_probs, page_names):
        page_votes[page].append(prob)

    correct = correct_top5 = 0
    for page, probs in page_votes.items():
        avg_prob = np.mean(probs, axis=0)
        true_writer = test_df[test_df['page'] == page].iloc[0]['writer_id']
        true_label = global_label_map[true_writer]
        if int(np.argmax(avg_prob)) == true_label:
            correct += 1
        if int(true_label) in np.argsort(-avg_prob)[:5]:
            correct_top5 += 1

    n_pages = len(page_votes)
    acc = correct / n_pages
    acc_top5 = correct_top5 / n_pages
    print(f"\n✅ 页面级准确率: {acc * 100:.2f}% ({correct}/{n_pages})  "
          f"| top-5: {acc_top5 * 100:.2f}%")
    print(f"   伪字符级: {char_correct / n_chars * 100:.2f}%  "
          f"| top-5: {char_correct_top5 / n_chars * 100:.2f}%")

    del test_dataset, test_loader
    gc.collect()
    if was_training:
        model.train()

    if return_details:
        return {
            'page_acc': acc,
            'page_acc_top5': acc_top5,
            'char_acc': char_correct / n_chars,
            'char_acc_top5': char_correct_top5 / n_chars,
            'n_pages': n_pages,
            'n_pages_correct': correct,
            'n_chars': n_chars,
        }
    return acc


# ============================================================
# 20-test：训练完成后调用，开 DropSegment
# ============================================================

# ============================================================
# 多模型共享测试集的评估（一次建集，多个模型一起前向）
# ============================================================

def evaluate_page_level_multi(models, metadata_file, global_label_map, device,
                              batch_size=128, num_workers=8,
                              page_workers=8, segment_workers=8,
                              apply_drop=False, needs_dynamic=None,
                              seq_len=200, dynamic_speed_clip=10.0,
                              use_bitmap=True, verbose=True):
    """
    多个模型**共享同一份测试集张量**的页面级评估。

    为什么值得共享
    --------------
    朴素做法是每个模型各调一次 `evaluate_page_level`，代价是：
      · 测试集**重建两次**（加载 1019 页 + 预切分 + 缓存 DropSegment 索引）
      · 测试张量**遍历两次**
    实测（RTX 4090 / 1019 页 / 299700 个伪字符）：每次 ≈1.5 分钟建集
    + 4.4 分钟推理，两个模型合计 ≈12 分钟/轮；80 轮光评估就 ~16 小时。

    共享之后建集只付一次，且两个模型看到**逐字节相同**的测试批次，
    对比结论里不会再混进任何数据侧差异。

    内存：只保留一份测试集，比顺序建两次更省（每次建集都有峰值）。

    Args:
        models: {名字: 已 eval() 的模型}
        needs_dynamic: {名字: bool}，缺省按 False（纯静态）

    Returns:
        {名字: {'page_acc','page_acc_top5','char_acc','char_acc_top5',
                'n_pages','n_pages_correct','n_chars'}}
    """
    names = list(models.keys())
    if not names:
        return {}
    if needs_dynamic is None:
        needs_dynamic = {n: False for n in names}
    any_dynamic = any(needs_dynamic.get(n, False) for n in names)

    was_training = {n: models[n].training for n in names}
    for n in names:
        models[n].eval()

    dataset = _make_dataset(metadata_file, apply_drop, any_dynamic,
                            page_workers, segment_workers,
                            seq_len, dynamic_speed_clip, use_bitmap=use_bitmap)
    loader = _make_loader(dataset, batch_size, num_workers)

    df_test = pd.read_csv(metadata_file, encoding='utf-8-sig')
    df_test['writer_id'] = df_test['writer_id'].astype(str)
    test_df = df_test[df_test['split'] == 'Test'].reset_index(drop=True)
    page_names = [test_df.iloc[pi]['page'] for pi, _, _ in dataset.samples]

    if verbose:
        print(f"\n📊 共享推理测试集（apply_drop={apply_drop}, "
              f"模型={names}, 只建一次数据集）...")

    probs_by_model = {n: [] for n in names}
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="并行推理", disable=not verbose):
            inputs, dynamic, labels = _unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            dyn = (dynamic.to(device, non_blocking=True)
                   if dynamic is not None else None)
            all_labels.append(labels.numpy())
            for n in names:
                use_dyn = dyn if needs_dynamic.get(n, False) else None
                out = _forward(models[n], inputs, use_dyn)
                probs_by_model[n].append(
                    torch.softmax(out, dim=1).detach().cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    n_chars = int(len(all_labels))

    # ---- 页面级投票（每个模型各自把页内所有伪字符的概率取平均）----
    results = {}
    for n in names:
        probs = np.concatenate(probs_by_model[n], axis=0)
        char_pred = probs.argmax(1)
        char_correct = int((char_pred == all_labels).sum())
        char_correct_top5 = _topk_correct(probs, all_labels, 5)

        votes = defaultdict(list)
        for prob, page in zip(probs, page_names):
            votes[page].append(prob)

        correct = correct_top5 = 0
        for page, plist in votes.items():
            avg = np.mean(plist, axis=0)
            true_writer = test_df[test_df['page'] == page].iloc[0]['writer_id']
            true_label = int(global_label_map[true_writer])
            if int(np.argmax(avg)) == true_label:
                correct += 1
            if true_label in np.argsort(-avg)[:5]:
                correct_top5 += 1
        n_pages = len(votes)
        acc = correct / n_pages
        acc_top5 = correct_top5 / n_pages
        results[n] = {
            'page_acc': acc,
            'page_acc_top5': acc_top5,
            'char_acc': char_correct / n_chars,
            'char_acc_top5': char_correct_top5 / n_chars,
            'n_pages': n_pages,
            'n_pages_correct': correct,
            'n_chars': n_chars,
        }
        if verbose:
            print(f"   {n:12s} page_acc={acc*100:6.2f}% ({correct}/{n_pages})"
                  f"  top5={acc_top5*100:6.2f}%"
                  f"  char_acc={char_correct/n_chars*100:6.2f}%")

    del dataset, loader, probs_by_model
    gc.collect()
    for n in names:
        if was_training[n]:
            models[n].train()
    return results


# ============================================================
# 20-test 增强评估（单模型，保留原有接口）
# ============================================================

def evaluate_page_level_20test(model, metadata_file, global_label_map, device,
                                batch_size=256, num_workers=8, n_test=20,
                                page_workers=8, segment_workers=8,
                                log_name='20test_records.csv',
                                apply_drop=True, return_dynamic=False,
                                seq_len=200, dynamic_speed_clip=10.0,
                                records_out=None):
    """
    Args:
        records_out: 若提供 list，则把每轮 (round, round_acc, cumulative_acc)
            追加进去，供配对 t 检验等统计分析使用（不改变返回值）。
    """
    was_training = model.training
    model.eval()

    # ========== 初始化元信息 ==========
    print(f"\n📋 初始化测试集元信息（apply_drop={apply_drop}, "
          f"return_dynamic={return_dynamic}）...")
    base_dataset = _make_dataset(metadata_file, apply_drop, return_dynamic,
                                 page_workers, segment_workers,
                                 seq_len, dynamic_speed_clip)
    sample_pages = list(base_dataset.page_names_for_samples)
    n_samples = len(sample_pages)
    unique_pages = sorted(set(sample_pages))
    print(f"✅ 测试集共 {n_samples} 个伪字符，分布在 {len(unique_pages)} 个页面。")

    page_to_true_label, _ = _build_test_meta(metadata_file, global_label_map,
                                             sample_pages)

    del base_dataset
    gc.collect()

    # ========== 累积概率 + 每轮记录 ==========
    accumulated_probs = None
    per_round_records = []

    for r in range(n_test):
        print(f"\n🔄 第 {r+1}/{n_test} 轮增强")
        dataset = _make_dataset(metadata_file, apply_drop, return_dynamic,
                                page_workers, segment_workers,
                                seq_len, dynamic_speed_clip)
        loader = _make_loader(dataset, batch_size, num_workers)

        round_probs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Round {r+1}"):
                inputs, dynamic, _ = _unpack_batch(batch)
                inputs = inputs.to(device, non_blocking=True)
                if dynamic is not None:
                    dynamic = dynamic.to(device, non_blocking=True)
                outputs = _forward(model, inputs, dynamic)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                round_probs.append(probs)
        round_probs = np.concatenate(round_probs, axis=0)

        round_page_acc = _compute_page_acc(round_probs, sample_pages,
                                           page_to_true_label)

        if accumulated_probs is None:
            accumulated_probs = round_probs.copy()
        else:
            accumulated_probs += round_probs

        cum_avg_probs = accumulated_probs / (r + 1)
        cumulative_page_acc = _compute_page_acc(cum_avg_probs, sample_pages,
                                                page_to_true_label)

        per_round_records.append((r + 1, round_page_acc, cumulative_page_acc))
        print(f"  → 第 {r+1} 轮 page acc: {round_page_acc*100:.2f}%  "
              f"| 累积 page acc: {cumulative_page_acc*100:.2f}%")

        del dataset, loader, round_probs
        gc.collect()
        torch.cuda.empty_cache()

    # ========== 保存 CSV ==========
    log_dir = 'outputs/logs'
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, log_name)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'round_page_acc', 'cumulative_page_acc'])
        for rec in per_round_records:
            writer.writerow([rec[0], f"{rec[1]:.4f}", f"{rec[2]:.4f}"])
    print(f"\n📝 每轮记录已保存至 {csv_path}")

    if records_out is not None:
        records_out.extend(per_round_records)

    final_acc = per_round_records[-1][2]
    print(f"\n✅ {n_test}-test 最终页面级准确率: {final_acc * 100:.2f}%")

    if was_training:
        model.train()
    return final_acc


# ============================================================
# 多模型共享增强的 20-test（配对评估）
# ============================================================

def evaluate_page_level_20test_multi(models, metadata_file, global_label_map,
                                     device, n_test=20, batch_size=256,
                                     num_workers=8, page_workers=8,
                                     segment_workers=8, apply_drop=True,
                                     needs_dynamic=None, seq_len=200,
                                     dynamic_speed_clip=10.0, use_bitmap=True,
                                     log_path=None, verbose=True):
    """
    多个模型**共享同一批增强样本**的 20-test。

    为什么要共享而不是各跑各的
    --------------------------
    20-test 的本质是「每次重新随机 DropSegment，看模型稳不稳」。
    如果模型 A 和模型 B 各自独立采样 20 次，它们吃的增强样本不同，
    两组的差异里就混进了「增强随机性」这个变量 —— 这样算出来的
    t 检验是**无效配对**。

    本函数每轮只构建**一个** DataLoader，所有模型前向同一批张量，于是：
      1. 模型之间严格配对，可以直接做配对 t 检验
      2. 数据管道的开销只付一次（本项目瓶颈之一）

    内存优化
    --------
    朴素实现要把全部样本概率留成 (299700, 1019) float32 ≈ 1.2 GB/模型，
    再累积一份就 2.4 GB/模型，两模型加起来 5 GB —— 在 60 GB 机器上
    叠加数据集缓存后很容易被 OOM killer 干掉。

    这里改成**按页面聚合**：每页只留一个 1019 维平均概率，
    总量 1019×1019 float64 ≈ 8 MB，内存降低约 300 倍。
    数学上等价：每轮每页的样本数不变，所以「先按页平均、再按轮平均」
    与原来的「先按轮累积、再按页平均」结果一致。样本数若变化会直接报错。

    Returns:
        {
          'n_test': int,
          'pages': int,
          'models': {name: {'per_round': [...], 'cumulative': [...],
                            'final_acc': float, 'mean_acc': float,
                            'std_acc': float}},
          'log_path': str or None,
        }
    """
    names = list(models.keys())
    if needs_dynamic is None:
        needs_dynamic = {n: False for n in names}
    any_dynamic = any(needs_dynamic.get(n, False) for n in names)

    was_training = {n: models[n].training for n in names}
    for n in names:
        models[n].eval()

    # ---- 测试集元信息（只建一次） ----
    print(f"\n📋 初始化 20-test 元信息（apply_drop={apply_drop}, "
          f"共享增强, 模型={names}）...")
    base = _make_dataset(metadata_file, apply_drop, any_dynamic,
                         page_workers, segment_workers,
                         seq_len, dynamic_speed_clip, use_bitmap=use_bitmap)
    sample_pages = list(base.page_names_for_samples)
    n_samples = len(sample_pages)
    unique_pages = sorted(set(sample_pages))
    page_idx = {p: i for i, p in enumerate(unique_pages)}
    page_to_true_label, _ = _build_test_meta(metadata_file, global_label_map,
                                             sample_pages)
    del base
    gc.collect()

    # 每页的真实标签向量（对齐到 unique_pages 顺序），用于向量化算准确率
    true_labels = np.array([page_to_true_label[p] for p in unique_pages])
    print(f"✅ {n_samples} 个伪字符 / {len(unique_pages)} 个页面 / "
          f"{n_test} 轮增强")

    n_pages, n_cls = len(unique_pages), len(global_label_map)
    page_pos = np.array([page_idx[p] for p in sample_pages])

    per_round = {n: [] for n in names}       # 每轮的**累积**页面级 acc
    round_indep = {n: [] for n in names}     # 每轮的**独立**页面级 acc
    acc_sum = {n: np.zeros((n_pages, n_cls), dtype=np.float64) for n in names}
    counts_ref = None

    for r in range(n_test):
        print(f"\n🔄 第 {r+1}/{n_test} 轮（所有模型吃同一批增强）")
        dataset = _make_dataset(metadata_file, apply_drop, any_dynamic,
                                page_workers, segment_workers,
                                seq_len, dynamic_speed_clip, use_bitmap=use_bitmap)
        loader = _make_loader(dataset, batch_size, num_workers)

        # 本轮每个模型的「每页概率和」
        sums = {n: np.zeros((n_pages, n_cls), dtype=np.float64) for n in names}
        counts = np.zeros(n_pages, dtype=np.int64)
        seen = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Round {r+1}", disable=not verbose):
                inputs, dynamic, _ = _unpack_batch(batch)
                inputs = inputs.to(device, non_blocking=True)
                dyn = (dynamic.to(device, non_blocking=True)
                       if dynamic is not None else None)
                bs = inputs.size(0)
                idx = page_pos[seen:seen + bs]
                seen += bs
                np.add.at(counts, idx, 1)
                for n in names:
                    use_dyn = dyn if needs_dynamic.get(n, False) else None
                    out = _forward(models[n], inputs, use_dyn)
                    probs = torch.softmax(out, dim=1).double().cpu().numpy()
                    for i in range(bs):
                        sums[n][idx[i]] += probs[i]

        if counts_ref is None:
            counts_ref = counts.copy()
        elif not np.array_equal(counts_ref, counts):
            raise RuntimeError(
                "20-test 各轮的伪字符数/页面分布不一致，无法按页聚合。"
                "请检查 DropSegment 是否改变了样本切分。")

        denom = np.maximum(counts, 1)[:, None]
        for n in names:
            page_mean = sums[n] / denom            # (pages, classes)
            # 本轮**独立**准确率：只用这一轮的 DropSegment 结果
            this_round = float((page_mean.argmax(1) == true_labels).mean())
            round_indep[n].append(this_round)
            acc_sum[n] += page_mean
            # 累积准确率：把前 r+1 轮的概率平均后再 argmax（论文口径）
            cum = acc_sum[n] / (r + 1)
            acc = float((cum.argmax(1) == true_labels).mean())
            per_round[n].append(acc)
            print(f"   {n:12s} 本轮 {this_round*100:6.2f}%"
                  f"  | 累积 {acc*100:6.2f}%")

        del dataset, loader, sums
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 统计 ----
    # per_round 记「累积到该轮」的准确率（论文口径：前 r 轮概率平均后再 argmax）；
    # round_indep 记「只用该轮」的独立准确率。
    #
    # ⚠️ 早期版本用 `indep_r = (r+1)*cum_r - r*cum_{r-1}` 反推独立值，那是**错的**：
    #    累积值是对概率平均后重新 argmax 得到的，与各轮准确率的算术平均没有线性
    #    关系。实测反推出来的"独立准确率"出现了 1.0196（>100%），由此算出的
    #    mean/std 也全部失真。现在直接记录真实的本轮准确率。
    result = {'n_test': n_test, 'pages': n_pages, 'models': {}}
    for n in names:
        cum = per_round[n]
        indep = round_indep[n]
        result['models'][n] = {
            'cumulative': cum,
            'per_round_acc': indep,
            'final_acc': cum[-1],
            'mean_acc': float(np.mean(indep)),
            'std_acc': float(np.std(indep, ddof=1)) if len(indep) > 1 else 0.0,
        }
        m = result['models'][n]
        print(f"\n✅ {n} {n_test}-test: 最终累积 {m['final_acc']*100:.2f}%  "
              f"| 单轮均值 {m['mean_acc']*100:.2f}% ± {m['std_acc']*100:.2f}%")

    # ---- 落盘 ----
    if log_path:
        _ensure_parent(log_path)
        header = ['round']
        for n in names:
            header += [f'{n}_round_acc', f'{n}_cumulative_acc']
        with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in range(n_test):
                row = [r + 1]
                for n in names:
                    row += [f"{result['models'][n]['per_round_acc'][r]:.6f}",
                            f"{result['models'][n]['cumulative'][r]:.6f}"]
                w.writerow(row)
        result['log_path'] = log_path
        print(f"📝 20-test 逐轮记录: {log_path}")

        # 追加到小结 CSV（mean±std，论文直接引用）
        summ = os.path.join(os.path.dirname(log_path), '20test_summary.csv')
        exists = os.path.exists(summ)
        with open(summ, 'a' if exists else 'w', newline='',
                  encoding='utf-8-sig') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(['run_name', 'n_test', 'pages', 'final_cum_acc',
                            'round_mean_acc', 'round_std_acc', 'finished_at'])
            for n in names:
                m = result['models'][n]
                w.writerow([n, n_test, n_pages, f"{m['final_acc']:.6f}",
                            f"{m['mean_acc']:.6f}", f"{m['std_acc']:.6f}",
                            time.strftime('%Y-%m-%d %H:%M:%S')])
        print(f"📝 20-test 汇总: {summ}")

    for n in names:
        if was_training[n]:
            models[n].train()
    return result


def _ensure_parent(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ============================================================
# 逐样本特征 / 概率导出（互补性分析用）
# ============================================================

def extract_test_features(model, metadata_file, global_label_map, device,
                          batch_size=256, num_workers=8,
                          page_workers=8, segment_workers=8,
                          return_dynamic=None, apply_drop=False,
                          seq_len=200, dynamic_speed_clip=10.0,
                          collect_features=True):
    """
    在**确定性测试集**（apply_drop=False）上导出逐样本信息。

    用于：
      - 5.2 互补性分析（静态特征 vs 动态特征做 CCA）
      - 5.3 误差分解（四象限：都对 / 仅静态对 / 仅动态对 / 都错）

    Args:
        return_dynamic: None 表示依据模型模式自动推断。
        collect_features: 是否额外收集 forward_features 的中间特征
            （dual 模式收集静态 512 维与动态 256 维；static/dynamic 模式
            分别只收集对应分支）。
        apply_drop: 强烈建议 False。逐样本对齐要求测试内容在多次运行间一致。

    Returns:
        dict:
            'probs'          (N, C) float32 逐样本 softmax 概率
            'labels'         (N,)   int64   逐样本真实作者标签
            'pages'          (N,)   object  逐样本所属页面
            'sample_pages'   (N,)   object  页面名（用于页面级投票）
            'static_feats'   (N, 512) 或 None
            'dynamic_feats'  (N, 256) 或 None
            'fused_feats'    (N, 768) 或 None
            'page_to_true_label' dict
    """
    if return_dynamic is None:
        return_dynamic = getattr(model, 'mode', 'static') in ('dynamic', 'dual')

    dataset = _make_dataset(metadata_file, apply_drop, return_dynamic,
                            page_workers, segment_workers,
                            seq_len, dynamic_speed_clip)
    loader = _make_loader(dataset, batch_size, num_workers)

    sample_pages = list(dataset.page_names_for_samples)
    page_to_true_label, sample_true_labels = _build_test_meta(
        metadata_file, global_label_map, sample_pages
    )

    was_training = model.training
    model.eval()

    all_probs, all_static, all_dynamic, all_fused = [], [], [], []
    has_forward_features = hasattr(model, 'forward_features')

    print(f"\n🔎 导出测试集逐样本信息（apply_drop={apply_drop}, "
          f"return_dynamic={return_dynamic}）...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="特征导出"):
            inputs, dynamic, _ = _unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            if dynamic is not None:
                dynamic = dynamic.to(device, non_blocking=True)

            outputs = _forward(model, inputs, dynamic)
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())

            if collect_features and has_forward_features:
                s_feat, d_feat = model.forward_features(inputs, dynamic)
                if s_feat is not None:
                    all_static.append(s_feat.cpu().numpy())
                if d_feat is not None:
                    all_dynamic.append(d_feat.cpu().numpy())
                if s_feat is not None and d_feat is not None:
                    all_fused.append(
                        torch.cat([s_feat, d_feat], dim=1).cpu().numpy()
                    )

    del dataset, loader
    gc.collect()
    if was_training:
        model.train()

    result = {
        'probs': np.concatenate(all_probs, axis=0).astype(np.float32),
        'labels': sample_true_labels,
        'pages': np.array(sample_pages, dtype=object),
        'sample_pages': sample_pages,
        'static_feats': (np.concatenate(all_static, axis=0).astype(np.float32)
                         if all_static else None),
        'dynamic_feats': (np.concatenate(all_dynamic, axis=0).astype(np.float32)
                          if all_dynamic else None),
        'fused_feats': (np.concatenate(all_fused, axis=0).astype(np.float32)
                        if all_fused else None),
        'page_to_true_label': page_to_true_label,
    }
    print(f"✅ 导出完成：{result['probs'].shape[0]} 个样本，"
          f"{len(page_to_true_label)} 个页面。")
    return result
