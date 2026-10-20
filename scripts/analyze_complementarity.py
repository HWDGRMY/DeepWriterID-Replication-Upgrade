"""
互补性分析：CCA + 误差四象限分解 + 配对 t 检验 + t-SNE 可视化。

对应方案文档的 5.1 / 5.2 / 5.3 与 6.2 节，用于回答审稿人最关心的三个问题：

  1. 「动态特征只是静态特征的重复吗？」
     → CCA：若前 10 个典型相关系数普遍 < 0.5，说明两个模态互补。
  2. 「融合的提升是真实存在还是噪声？」
     → 配对 t 检验（20-test 每轮准确率配对），报告 p 值。
  3. 「哪些样本静态失效但动态有效？」
     → 误差四象限分解：都对 / 仅静态对 / 仅动态对 / 都错。
       只要 `dynamic_only > 0`，就存在「静态失效但动态有效」的样本，
       这是互补性最直接的证据。

用法
----
    # 三组实验都训练完之后
    python scripts/analyze_complementarity.py \
        --static_ckpt  outputs/checkpoints/static_stream_best.pth \
        --dynamic_ckpt outputs/checkpoints/dynamic_stream_best.pth \
        --dual_ckpt    outputs/checkpoints/dual_stream_best.pth

    # 只做单模型的特征分析
    python scripts/analyze_complementarity.py --dual_ckpt outputs/checkpoints/dual_stream_best.pth

输出（默认落在 outputs/analysis/）
----
    cca_correlations.csv        典型相关系数
    error_quadrants.csv         四象限计数
    error_quadrants.png         四象限柱状图
    tsne_static_dynamic.png     静态 / 动态特征空间 t-SNE 对比
    significance.csv            配对 t 检验结果
    summary.md                  汇总报告（可直接贴进论文）

重要前提
--------
逐样本误差分解与 CCA 要求**测试内容在多次运行间一致**，因此本脚本固定使用
`apply_drop=False` 的确定性测试集。页面级准确率仍按页面投票计算。
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings

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

import matplotlib                                        # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402

from src.models.dual_stream import DualStreamWriterID    # noqa: E402
from src.evaluation.evaluator import (                   # noqa: E402
    extract_test_features,
    evaluate_page_level_20test,
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def load_model(ckpt_path, mode, num_classes, device, verbose=True):
    """
    按模式构造模型并加载检查点。

    **架构超参从检查点里保存的 config 读取**，而不是用脚本里的默认值。
    否则一旦做过 d_model / num_layers / fusion_hidden / pooling 的消融扫描，
    分析脚本就会因为形状不匹配而崩溃（或者更糟：静默加载错模型）。

    同时校验类别数：训练与分析必须使用同一份 metadata 划分，
    否则标签索引对不上，所有对比结论都会失效。
    """
    ckpt = None
    cfg = {}
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and isinstance(ckpt.get('config'), dict):
            cfg = ckpt['config']

    model = DualStreamWriterID(
        num_classes=num_classes,
        mode=mode,
        static_dim=cfg.get('feature_dim', 512),
        in_chans=cfg.get('in_chans', 64),
        drop_path_rate=0.0,          # 推理不需要随机正则
        dropout_head=0.0,
        dropout_classifier=0.0,
        dynamic_dim=cfg.get('dynamic_dim', 256),
        seq_len=cfg.get('seq_len', 200),
        dynamic_in_dim=cfg.get('dynamic_in_dim', 2),
        d_model=cfg.get('d_model', 64),
        nhead=cfg.get('nhead', 4),
        num_layers=cfg.get('num_layers', 4),
        dim_feedforward=cfg.get('dim_feedforward', 256),
        dynamic_dropout=0.0,
        pooling=cfg.get('pooling', 'attention'),
        pos_encoding=cfg.get('pos_encoding', 'sinusoidal'),
        fusion_hidden=cfg.get('fusion_hidden', 512),
        fusion_dropout=0.0,
        freeze_static=False,
    )

    if ckpt is None:
        print(f"⚠️ 未找到 {mode} 检查点 ({ckpt_path})，使用随机初始化权重"
              f"（结果仅供流程验证）")
        return model.to(device).eval()

    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    # ---- 类别数一致性校验 ----
    for key, tensor in state.items():
        if key.endswith('classifier.1.weight') or key == 'dynamic_classifier.1.weight' \
                or key.endswith('fusion.4.weight'):
            ckpt_classes = int(tensor.shape[0])
            if ckpt_classes != num_classes:
                raise RuntimeError(
                    f"\n类别数不匹配：检查点 {os.path.basename(ckpt_path)} 是 "
                    f"{ckpt_classes} 类，而当前 metadata 解析出 {num_classes} 类。\n"
                    f"训练与互补性分析必须使用**同一份 metadata 划分**，"
                    f"否则标签索引对不上、所有对比结论都会失效。\n"
                    f"请检查 --metadata 是否与训练时一致。"
                )
            break

    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose:
        print(f"✅ 已加载 {mode} 模型 ← {ckpt_path}")
        if cfg:
            print(f"   架构(取自检查点): d_model={cfg.get('d_model', 64)}, "
                  f"num_layers={cfg.get('num_layers', 4)}, "
                  f"dim_feedforward={cfg.get('dim_feedforward', 256)}, "
                  f"dynamic_dim={cfg.get('dynamic_dim', 256)}, "
                  f"fusion_hidden={cfg.get('fusion_hidden', 512)}, "
                  f"pooling={cfg.get('pooling', 'attention')}")
            print(f"   特征版本: {cfg.get('static_feature_version')}")
        if isinstance(ckpt, dict) and 'best_acc' in ckpt:
            print(f"   来源 best_acc={ckpt['best_acc']}")
        core_missing = [k for k in missing if 'dynamic_branch' not in k
                        and 'fusion' not in k and 'dynamic_classifier' not in k]
        if core_missing:
            print(f"   ⚠️ 有 {len(core_missing)} 个核心权重未匹配: "
                  f"{core_missing[:3]}")

    return model.to(device).eval()


def page_level_acc(probs, sample_pages, page_to_true_label):
    from collections import defaultdict
    votes = defaultdict(list)
    for i, page in enumerate(sample_pages):
        votes[page].append(probs[i])
    correct = 0
    for page, ps in votes.items():
        if int(np.argmax(np.mean(ps, axis=0))) == page_to_true_label[page]:
            correct += 1
    return correct / max(len(votes), 1)


# ---------------------------------------------------------------------------
# 5.2 典型相关分析
# ---------------------------------------------------------------------------

def run_cca(static_feats, dynamic_feats, n_components=10, out_dir='outputs/analysis'):
    """
    计算静态特征与动态特征之间的典型相关系数。

    相关系数普遍偏低（< 0.5）说明两个模态携带的信息大部分不重叠，
    即动态特征不是静态特征的简单重复。
    """
    print("\n" + "=" * 74)
    print("【5.2 互补性分析】典型相关分析（CCA）")
    print("=" * 74)

    if static_feats is None or dynamic_feats is None:
        print("⚠️ 缺少静态或动态特征，跳过 CCA")
        return None

    from sklearn.cross_decomposition import CCA

    # 特征标准化（CCA 对尺度敏感）
    X = (static_feats - static_feats.mean(0)) / (static_feats.std(0) + 1e-8)
    Y = (dynamic_feats - dynamic_feats.mean(0)) / (dynamic_feats.std(0) + 1e-8)

    # 高维小样本时 CCA 容易过拟合，先做 PCA 降维
    n_samples = X.shape[0]
    max_comp = min(n_components, X.shape[1], Y.shape[1], n_samples - 1)
    if max_comp < 1:
        print("⚠️ 样本量不足以做 CCA，跳过")
        return None

    from sklearn.decomposition import PCA
    n_pca = min(64, X.shape[0] - 1, X.shape[1])
    if X.shape[1] > n_pca:
        X = PCA(n_components=n_pca, random_state=0).fit_transform(X)
    n_pca_y = min(64, Y.shape[0] - 1, Y.shape[1])
    if Y.shape[1] > n_pca_y:
        Y = PCA(n_components=n_pca_y, random_state=0).fit_transform(Y)

    max_comp = min(max_comp, X.shape[1], Y.shape[1])
    if max_comp < 1:
        print("⚠️ 降维后维度不足，跳过 CCA")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cca = CCA(n_components=max_comp, max_iter=1000)
        X_c, Y_c = cca.fit_transform(X, Y)

    correlations = []
    for i in range(max_comp):
        r = float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1])
        correlations.append(0.0 if np.isnan(r) else r)

    print(f"\n前 {max_comp} 个典型相关系数:")
    for i, r in enumerate(correlations):
        print(f"   CC{i+1:<3} = {r:.4f}")

    mean_cc = float(np.mean(correlations))
    max_cc = float(np.max(correlations))
    print(f"\n   平均典型相关 = {mean_cc:.4f}")
    print(f"   最大典型相关 = {max_cc:.4f}")
    if max_cc < 0.5:
        print("   ✅ 最大典型相关 < 0.5 → 两个模态高度互补，"
              "动态特征不是静态特征的重复")
    elif max_cc < 0.8:
        print("   ⚠️ 最大典型相关在 [0.5, 0.8) → 部分重叠但仍有互补信息")
    else:
        print("   ❌ 最大典型相关 >= 0.8 → 两个模态高度冗余，"
              "需要重新设计动态特征")

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        'component': list(range(1, max_comp + 1)),
        'canonical_correlation': correlations,
    }).to_csv(os.path.join(out_dir, 'cca_correlations.csv'),
              index=False, encoding='utf-8-sig')

    return {'correlations': correlations, 'mean': mean_cc, 'max': max_cc}


# ---------------------------------------------------------------------------
# 5.3 误差四象限分解
# ---------------------------------------------------------------------------

def run_error_decomposition(res_a, res_b, res_c=None,
                            name_a='static', name_b='dynamic', name_c='dual',
                            out_dir='outputs/analysis'):
    """
    逐样本比较两个（可选三个）模型的预测，统计四象限。

    **两个模型怎么选，决定了论证的强度：**

    · 若把「纯动态分支」作为 B 一起训练：A=static, B=dynamic, C=dual。
      `dynamic_only > 0` 说明存在「静态失效、动态有效」的样本。
    · 若不做纯动态实验（三组实验方案 A0/A1/C）：
      应当把 **A=A1（修正后的最强静态基线）** 与 **B=C（双路融合）** 对比。
      此时 `b_only > 0` 的含义更强——「**最好的纯静态模型都错了，而融合对了**」，
      这是比「击败一个有缺陷的基线」硬得多的互补性证据。

    注意：这里用的是**逐样本（伪字符级）**预测，而不是页面级投票，
    因为逐样本对齐才能定位「静态失效但融合有效」的具体形态。
    要求多个模型使用**同一个确定性测试集**（apply_drop=False），
    否则样本顺序不一致、无法对齐。
    """
    print("\n" + "=" * 74)
    print(f"【5.3 误差分析】逐样本四象限分解: {name_a} vs {name_b}")
    print("=" * 74)

    if res_a is None or res_b is None:
        print(f"⚠️ 缺少 {name_a} 或 {name_b} 的推理结果，跳过误差分解")
        return None

    labels = res_a['labels']
    if not np.array_equal(labels, res_b['labels']):
        print(f"❌ {name_a} 与 {name_b} 的样本顺序不一致，无法逐样本对齐。")
        print("   请确保两者使用相同的 metadata、apply_drop=False "
              "以及相同的特征编码版本。")
        return None

    pred_a = res_a['probs'].argmax(1)
    pred_b = res_b['probs'].argmax(1)
    pred_c = res_c['probs'].argmax(1) if res_c is not None else None

    correct_a = pred_a == labels
    correct_b = pred_b == labels

    quad = {
        'both_correct': int(np.sum(correct_a & correct_b)),
        f'{name_a}_only': int(np.sum(correct_a & ~correct_b)),
        f'{name_b}_only': int(np.sum(~correct_a & correct_b)),
        'both_wrong': int(np.sum(~correct_a & ~correct_b)),
    }
    total = int(len(labels))
    a_only_key = f'{name_a}_only'
    b_only_key = f'{name_b}_only'

    print(f"\n逐样本统计（共 {total} 个伪字符）:")
    print(f"   都对            both_correct = {quad['both_correct']:>7d} "
          f"({quad['both_correct']/total*100:5.2f}%)")
    print(f"   仅 {name_a:<8} 对 {a_only_key:<16} = {quad[a_only_key]:>7d} "
          f"({quad[a_only_key]/total*100:5.2f}%)")
    print(f"   仅 {name_b:<8} 对 {b_only_key:<16} = {quad[b_only_key]:>7d} "
          f"({quad[b_only_key]/total*100:5.2f}%)")
    print(f"   都错            both_wrong   = {quad['both_wrong']:>7d} "
          f"({quad['both_wrong']/total*100:5.2f}%)")

    print(f"\n   {name_a:<10} 逐样本准确率 = {correct_a.mean()*100:.2f}%")
    print(f"   {name_b:<10} 逐样本准确率 = {correct_b.mean()*100:.2f}%")
    if pred_c is not None:
        print(f"   {name_c:<10} 逐样本准确率 = {(pred_c == labels).mean()*100:.2f}%")

    if quad[b_only_key] > 0:
        print(f"\n   ✅ {b_only_key} = {quad[b_only_key]} > 0 → "
              f"存在「{name_a} 失效但 {name_b} 有效」的样本，互补性证据成立")
    else:
        print(f"\n   ❌ {b_only_key} = 0 → 未找到 {name_a} 失效而 {name_b} 有效的样本，"
              "互补性证据不足")

    if pred_c is not None:
        above = int(np.sum((pred_c == labels) & ~correct_a))
        below = int(np.sum((pred_c != labels) & correct_a))
        print(f"\n   {name_c} 相对 {name_a}：修正 {above} 个，新错 {below} 个，"
              f"净变化 {above - below:+d}")

    os.makedirs(out_dir, exist_ok=True)
    rows = [{'quadrant': k, 'count': v, 'ratio': v / total}
            for k, v in quad.items()]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'error_quadrants.csv'),
                              index=False, encoding='utf-8-sig')

    # 柱状图
    try:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        keys = ['both_correct', a_only_key, b_only_key, 'both_wrong']
        names = [f'both correct', f'{name_a} only', f'{name_b} only', 'both wrong']
        vals = [quad[k] for k in keys]
        colors = ['#4C9F70', '#3B7DD8', '#E8A33D', '#C94C4C']
        bars = ax.bar(names, vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v}\n({v/total*100:.1f}%)", ha='center', va='bottom',
                    fontsize=9)
        ax.set_ylabel('Number of pseudo-characters')
        ax.set_title(f'Error decomposition: {name_a} vs {name_b}')
        ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'error_quadrants.png'), dpi=200)
        plt.close(fig)
        print(f"   📊 四象限图已保存: {os.path.join(out_dir, 'error_quadrants.png')}")
    except Exception as exc:                          # noqa: BLE001
        print(f"   ⚠️ 四象限图绘制失败: {exc}")

    quad['_keys'] = {'a_only': a_only_key, 'b_only': b_only_key}
    return quad


# ---------------------------------------------------------------------------
# t-SNE 可视化
# ---------------------------------------------------------------------------

def run_tsne(static_feats, dynamic_feats, labels, out_dir='outputs/analysis',
             max_samples=3000, perplexity=30.0):
    """把静态 / 动态特征空间并排降维展示，观察两者的结构差异。"""
    print("\n" + "=" * 74)
    print("【6.2 可视化】静态 / 动态特征空间 t-SNE")
    print("=" * 74)

    if static_feats is None or dynamic_feats is None:
        print("⚠️ 缺少特征，跳过 t-SNE")
        return None

    from sklearn.manifold import TSNE

    n = len(labels)
    if n > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_samples, replace=False)
    else:
        idx = np.arange(n)

    Xs, Xd, y = static_feats[idx], dynamic_feats[idx], labels[idx]
    perp = float(min(perplexity, max(5.0, (len(idx) - 1) / 3.0)))

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (name, X) in zip(axes, [('Static (512-d)', Xs),
                                    ('Dynamic (256-d)', Xd)]):
        # 先 PCA 到 50 维，t-SNE 更快更稳
        from sklearn.decomposition import PCA
        n_pca = min(50, X.shape[0] - 1, X.shape[1])
        Xp = PCA(n_components=n_pca, random_state=0).fit_transform(X) \
            if X.shape[1] > n_pca else X
        emb = TSNE(n_components=2, perplexity=perp, init='pca',
                   random_state=0, max_iter=1000).fit_transform(Xp)
        results[name] = emb
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab20', s=4, alpha=0.7)
        ax.set_title(f'{name} — t-SNE')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'tsne_static_dynamic.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"   📊 t-SNE 图已保存: {path}")
    return results


# ---------------------------------------------------------------------------
# 配对 t 检验
# ---------------------------------------------------------------------------

def run_paired_20test(models, metadata_file, global_label_map, device,
                      n_test=20, batch_size=128, num_workers=8,
                      page_workers=8, segment_workers=8, seq_len=200,
                      dynamic_speed_clip=10.0, log_dir='outputs/logs'):
    """
    在**完全相同**的 20 组增强测试集上评测所有模型。

    为什么不能分别跑两次 20-test 再配对：
        `evaluate_page_level_20test` 每轮会重新做一次随机 DropSegment。
        两次独立运行之间，第 i 轮的增强内容并不相同，把它们的逐轮准确率
        配成对子在统计上是错的（虽然偏保守，但审稿人会质疑）。
        本函数让所有模型在同一轮里吃同一个 batch，配对才真正成立。

    副作用：数据集每轮只构建一次（而不是每个模型一次），总开销更低。

    Returns:
        dict: {mode: {'round_acc': [...], 'cumulative_acc': [...],
                      'final_acc': float}}
    """
    import gc as _gc
    from collections import defaultdict            # noqa: F401
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from src.evaluation.evaluator import (
        _make_dataset, _unpack_batch, _forward, _compute_page_acc,
        _build_test_meta, _loader_context,
    )

    names = list(models.keys())

    # 页面划分是确定性的，用一次 apply_drop=False 的构建拿到样本顺序
    base = _make_dataset(metadata_file, False, False, page_workers,
                         segment_workers, seq_len, dynamic_speed_clip)
    sample_pages = list(base.page_names_for_samples)
    page_to_true_label, _ = _build_test_meta(metadata_file, global_label_map,
                                             sample_pages)
    del base
    _gc.collect()

    modes = {n: getattr(models[n], 'mode', 'static') for n in names}
    needs_dynamic = {n: modes[n] != 'static' for n in names}
    any_dynamic = any(needs_dynamic.values())

    per_round = {n: [] for n in names}
    cumulative = {n: [] for n in names}
    accum = {n: None for n in names}

    print(f"\n📊 共享增强 {n_test}-test（所有模型吃同一批数据，配对有效）")
    for r in range(n_test):
        ds = _make_dataset(metadata_file, True, any_dynamic, page_workers,
                           segment_workers, seq_len, dynamic_speed_clip)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            multiprocessing_context=_loader_context(num_workers))

        round_probs = {n: [] for n in names}
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Round {r+1}/{n_test}"):
                inputs, dynamic, _ = _unpack_batch(batch)
                inputs = inputs.to(device, non_blocking=True)
                dyn_dev = (dynamic.to(device, non_blocking=True)
                           if dynamic is not None else None)
                for n in names:
                    d = dyn_dev if needs_dynamic[n] else None
                    out = _forward(models[n], inputs, d)
                    round_probs[n].append(
                        torch.softmax(out, dim=1).cpu().numpy()
                    )

        line = [f"  第 {r+1:>2}/{n_test} 轮:"]
        for n in names:
            rp = np.concatenate(round_probs[n], axis=0)
            acc = _compute_page_acc(rp, sample_pages, page_to_true_label)
            per_round[n].append(acc)
            accum[n] = rp.copy() if accum[n] is None else accum[n] + rp
            cum = _compute_page_acc(accum[n] / (r + 1), sample_pages,
                                    page_to_true_label)
            cumulative[n].append(cum)
            line.append(f"{n}={acc*100:5.2f}%")
        print("  ".join(line))

        del ds, loader, round_probs
        _gc.collect()
        torch.cuda.empty_cache()

    result = {}
    for n in names:
        result[n] = {
            'round_acc': per_round[n],
            'cumulative_acc': cumulative[n],
            'final_acc': cumulative[n][-1],
        }
        print(f"  → {n:<8} 最终累积页面级准确率 = {cumulative[n][-1]*100:.2f}%")

    # 落盘，便于复查（写进分析输出目录，避免依赖可能不可写的 outputs/）
    try:
        os.makedirs(log_dir, exist_ok=True)
        rows = []
        for i in range(n_test):
            row = {'round': i + 1}
            for n in names:
                row[f'{n}_round_acc'] = per_round[n][i]
                row[f'{n}_cumulative_acc'] = cumulative[n][i]
            rows.append(row)
        rec_path = os.path.join(log_dir, 'paired_20test_records.csv')
        pd.DataFrame(rows).to_csv(rec_path, index=False, encoding='utf-8-sig')
        print(f"  📝 逐轮记录已保存: {rec_path}")
    except Exception as exc:                          # noqa: BLE001
        print(f"  ⚠️ 逐轮记录保存失败（不影响后续分析）: {exc}")

    return result


def _to_acc_array(records):
    """兼容两种记录格式：[(round, acc, cum), ...] 或 [acc, ...]。"""
    if not records:
        return None
    if isinstance(records[0], (tuple, list)):
        return np.array([r[1] for r in records], dtype=np.float64)
    return np.array(records, dtype=np.float64)


def run_significance(static_records, dual_records, dynamic_records=None,
                     baseline_records=None, out_dir='outputs/analysis'):
    """
    对 20-test 的每轮页面级准确率做配对 t 检验。

    配对成立的前提：两个模型在同一轮看到的是**完全相同的增强内容**。
    这由 `run_paired_20test` 保证（所有模型吃同一个 batch），
    不要改用两次独立的 `evaluate_page_level_20test` 结果来配对。
    """
    print("\n" + "=" * 74)
    print("【5.1 显著性】20-test 每轮准确率配对 t 检验")
    print("=" * 74)

    if not static_records or not dual_records:
        print("⚠️ 缺少 20-test 逐轮记录，跳过显著性检验")
        return None

    from scipy import stats

    s = _to_acc_array(static_records)
    u = _to_acc_array(dual_records)
    n = min(len(s), len(u))
    s, u = s[:n], u[:n]

    diff = u - s
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0

    if n < 2 or not np.isfinite(sd) or sd < 1e-12:
        # 差值完全没有方差时配对 t 检验无定义（常见于类别数极少的退化场景）
        print(f"\n   配对数 = {n}")
        print(f"   静态  每轮均值 = {s.mean()*100:.2f}%")
        print(f"   融合  每轮均值 = {u.mean()*100:.2f}%")
        print(f"   平均提升 = {diff.mean()*100:+.2f} 个百分点")
        print("   ⚠️ 每轮差值方差为 0（两个模型逐轮结果完全相同），"
              "配对 t 检验无定义。")
        print("      常见原因：类别数过少或模型未真正训练，"
              "请在完整的 1019 类实验上运行本分析。")
        t_stat, p_value = float('nan'), float('nan')
    else:
        t_stat, p_value = stats.ttest_rel(u, s)

        print(f"\n   配对数 = {n}")
        print(f"   静态  每轮均值 = {s.mean()*100:.2f}% ± {s.std(ddof=1)*100:.2f}%")
        print(f"   融合  每轮均值 = {u.mean()*100:.2f}% ± {u.std(ddof=1)*100:.2f}%")
        print(f"   平均提升 = {diff.mean()*100:+.2f} 个百分点")
        print(f"   t = {t_stat:.4f},  p = {p_value:.6f}")

        if p_value < 0.05:
            print(f"   ✅ p < 0.05 → 融合带来的提升具有统计显著性")
        else:
            print(f"   ❌ p >= 0.05 → 提升不显著，需要更多轮次或检查融合设计")

    # Cohen's d（配对）
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    print(f"   Cohen's d = {d:.4f}")
    if abs(d) >= 0.8:
        print("   → 效应量：大")
    elif abs(d) >= 0.5:
        print("   → 效应量：中")
    elif abs(d) >= 0.2:
        print("   → 效应量：小")
    else:
        print("   → 效应量：可忽略")

    rows = [{
        'comparison': 'dual_vs_static', 'n_pairs': n,
        'static_mean': float(s.mean()), 'dual_mean': float(u.mean()),
        'mean_gain': float(diff.mean()),
        't_statistic': float(t_stat), 'p_value': float(p_value),
        'cohens_d': d,
    }]

    if dynamic_records:
        dyr = _to_acc_array(dynamic_records)[:n]
        d_diff = dyr - s
        d_sd = float(d_diff.std(ddof=1)) if n > 1 else 0.0
        if n < 2 or not np.isfinite(d_sd) or d_sd < 1e-12:
            t2, p2 = float('nan'), float('nan')
            print(f"\n   纯动态 vs 静态: 差值方差为 0，t 检验无定义，"
                  f"动态均值 = {dyr.mean()*100:.2f}%")
        else:
            t2, p2 = stats.ttest_rel(dyr, s)
            print(f"\n   纯动态 vs 静态: t = {t2:.4f}, p = {p2:.6f}, "
                  f"动态均值 = {dyr.mean()*100:.2f}%")
        rows.append({
            'comparison': 'dynamic_vs_static', 'n_pairs': n,
            'static_mean': float(s.mean()), 'dual_mean': float(dyr.mean()),
            'mean_gain': float(d_diff.mean()),
            't_statistic': float(t2), 'p_value': float(p2),
            'cohens_d': float(d_diff.mean() / (d_sd + 1e-12)),
        })

    os.makedirs(out_dir, exist_ok=True)

    # A1 vs A0：静态流水线修复本身是否带来显著提升
    if baseline_records:
        base = _to_acc_array(baseline_records)[:n]
        b_diff = s - base                      # s 是 A1（更强的静态基线）
        b_sd = float(b_diff.std(ddof=1)) if n > 1 else 0.0
        if n < 2 or not np.isfinite(b_sd) or b_sd < 1e-12:
            t3, p3 = float('nan'), float('nan')
            print(f"\n   A1 vs A0（静态流水线修复）: 差值方差为 0，t 检验无定义")
        else:
            t3, p3 = stats.ttest_rel(s, base)
            print(f"\n   A1 vs A0（静态流水线修复）: t = {t3:.4f}, p = {p3:.6f}, "
                  f"平均提升 = {b_diff.mean()*100:+.2f}pp")
            if p3 < 0.05:
                print(f"   ✅ p < 0.05 → 静态流水线修复带来的提升具有统计显著性")
            else:
                print(f"   ❌ p >= 0.05 → 修复带来的提升不显著")
        rows.append({
            'comparison': 'static_a1_vs_a0', 'n_pairs': n,
            'static_mean': float(base.mean()), 'dual_mean': float(s.mean()),
            'mean_gain': float(b_diff.mean()),
            't_statistic': float(t3), 'p_value': float(p3),
            'cohens_d': float(b_diff.mean() / (b_sd + 1e-12)),
        })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'significance.csv'),
                              index=False, encoding='utf-8-sig')
    return rows


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', type=str, default='data/features/metadata.csv')
    ap.add_argument('--static_ckpt', type=str,
                    default='outputs/checkpoints/static_stream_best.pth',
                    help='最强的纯静态基线，三组实验方案里应指向 A1')
    ap.add_argument('--dynamic_ckpt', type=str,
                    default='outputs/checkpoints/dynamic_stream_best.pth',
                    help='纯动态分支（可选；不做实验 B 时留空/不存在即可）')
    ap.add_argument('--dual_ckpt', type=str,
                    default='outputs/checkpoints/dual_stream_best.pth',
                    help='双路融合模型 C')
    ap.add_argument('--baseline_ckpt', type=str, default=None,
                    help='可选：A0（v2.0 原样的静态基线），'
                         '用于额外给出「A1 vs A0」的配对检验')
    ap.add_argument('--out_dir', type=str, default='outputs/analysis')
    ap.add_argument('--seq_len', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--page_workers', type=int, default=8)
    ap.add_argument('--segment_workers', type=int, default=8)
    ap.add_argument('--n_test', type=int, default=20,
                    help='20-test 轮数，用于配对 t 检验；设 0 跳过')
    ap.add_argument('--tsne_samples', type=int, default=3000)
    ap.add_argument('--skip_tsne', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 设备: {device}")

    df = pd.read_csv(args.metadata, encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)
    all_writers = sorted(df['writer_id'].unique())
    global_label_map = {w: i for i, w in enumerate(all_writers)}
    num_classes = len(global_label_map)
    print(f"📊 类别数: {num_classes}")
    if num_classes < 50:
        print("⚠️ 类别数过少，CCA / 误差分解 / t 检验的结果不具统计意义，"
              "仅供流程验证。\n"
              "   正式实验请使用完整的 data/features/metadata.csv（1019 类）。")

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  page_workers=args.page_workers,
                  segment_workers=args.segment_workers,
                  seq_len=args.seq_len, apply_drop=False)

    # ========== 确定性测试集上逐样本导出 ==========
    results = {}
    models = {}
    plan = [('static', args.static_ckpt),
            ('dynamic', args.dynamic_ckpt),
            ('dual', args.dual_ckpt)]
    if args.baseline_ckpt:
        # A0 也是纯静态模型，用 'static' 模式加载
        plan.append(('baseline', args.baseline_ckpt))

    for mode, ckpt in plan:
        if not os.path.exists(ckpt):
            print(f"\n⏭️  跳过 {mode}: 找不到 {ckpt}")
            continue
        load_mode = 'static' if mode == 'baseline' else mode
        model = load_model(ckpt, load_mode, num_classes, device)
        models[mode] = model
        print(f"\n🔎 导出 {mode} 模型的逐样本预测 / 特征 ...")
        results[mode] = extract_test_features(
            model, args.metadata, global_label_map, device, **common
        )

    if not results:
        print("\n❌ 没有可分析的模型，退出。")
        return

    res_static = results.get('static')
    res_dynamic = results.get('dynamic')
    res_dual = results.get('dual')

    # ========== 5.2 CCA ==========
    cca_res = None
    feature_source = res_dual or res_static or res_dynamic
    if feature_source is not None:
        cca_res = run_cca(feature_source['static_feats'],
                          feature_source['dynamic_feats'],
                          out_dir=args.out_dir)

    # ========== 5.3 误差四象限 ==========
    # 三组实验（A0/A1/C，不做纯动态）时，对比对象应当是
    # 「最强的纯静态基线 A1」 vs 「双路融合 C」——这比对比一个有缺陷的基线硬得多。
    if res_dynamic is not None:
        pair = (res_static, 'static', res_dynamic, 'dynamic')
    else:
        pair = (res_static, 'static', res_dual, 'dual')
    quad = run_error_decomposition(pair[0], pair[2], res_dual if res_dynamic else None,
                                   name_a=pair[1], name_b=pair[3],
                                   name_c='dual',
                                   out_dir=args.out_dir)

    # ========== 页面级准确率（确定性测试集） ==========
    print("\n" + "=" * 74)
    print("【逐样本模型准确率 / 页面级投票准确率】（确定性测试集）")
    print("=" * 74)
    page_accs = {}
    for mode, res in results.items():
        sample_labels = res['labels']
        char_acc = float((res['probs'].argmax(1) == sample_labels).mean())
        pacc = page_level_acc(res['probs'], res['sample_pages'],
                              res['page_to_true_label'])
        page_accs[mode] = pacc
        print(f"   {mode:<8} 伪字符级 = {char_acc*100:6.2f}%   "
              f"页面级 = {pacc*100:6.2f}%")

    # ========== t-SNE ==========
    if not args.skip_tsne and feature_source is not None:
        run_tsne(feature_source['static_feats'], feature_source['dynamic_feats'],
                 feature_source['labels'], out_dir=args.out_dir,
                 max_samples=args.tsne_samples)

    # ========== 配对 t 检验（20-test，共享增强） ==========
    sig = None
    if args.n_test > 0 and len(models) >= 2:
        print(f"\n{'='*74}\n【20-test 配对 t 检验】（{args.n_test} 轮，含 DropSegment，"
              f"所有模型共享同一组增强）\n{'='*74}")
        paired = run_paired_20test(
            models, args.metadata, global_label_map, device,
            n_test=args.n_test, batch_size=args.batch_size,
            num_workers=args.num_workers,
            page_workers=args.page_workers,
            segment_workers=args.segment_workers,
            seq_len=args.seq_len,
            log_dir=args.out_dir,
        )
        sig = run_significance(
            paired.get('static', {}).get('round_acc'),
            paired.get('dual', {}).get('round_acc'),
            paired.get('dynamic', {}).get('round_acc'),
            baseline_records=paired.get('baseline', {}).get('round_acc'),
            out_dir=args.out_dir,
        )
        # 20-test 成绩并入汇总
        for mode, res in paired.items():
            page_accs[f'{mode}_20test'] = res['final_acc']

    # ========== 汇总报告 ==========
    write_summary(args.out_dir, cca_res, quad, sig, page_accs, results)

    print("\n" + "=" * 74)
    print(f"🏁 分析完成，结果目录: {os.path.abspath(args.out_dir)}")
    print("=" * 74)


def write_summary(out_dir, cca_res, quad, sig, page_accs, results):
    """输出可直接贴进论文的汇总 markdown。"""
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# 伪时序双路融合 — 互补性分析汇总", ""]

    lines.append("## 各模式准确率（确定性测试集，apply_drop=False）")
    lines.append("")
    lines.append("| 模式 | 页面级准确率 |")
    lines.append("| :--- | ---: |")
    for mode, acc in page_accs.items():
        lines.append(f"| {mode} | {acc*100:.2f}% |")
    lines.append("")

    if cca_res:
        lines.append("## CCA 典型相关系数（5.2）")
        lines.append("")
        lines.append(f"- 前 {len(cca_res['correlations'])} 个典型相关系数: "
                     + ", ".join(f"{c:.4f}" for c in cca_res['correlations']))
        lines.append(f"- 平均 = {cca_res['mean']:.4f}，最大 = {cca_res['max']:.4f}")
        verdict = ("两个模态高度互补" if cca_res['max'] < 0.5
                   else "部分重叠但仍互补" if cca_res['max'] < 0.8
                   else "高度冗余")
        lines.append(f"- 结论: **{verdict}**")
        lines.append("")

    if quad:
        total = sum(v for k, v in quad.items() if not k.startswith('_'))
        keys = quad.get('_keys', {})
        a_key = keys.get('a_only', 'static_only')
        b_key = keys.get('b_only', 'dynamic_only')
        lines.append("## 误差四象限分解（5.3）")
        lines.append("")
        lines.append(f"对比对象: **{a_key.replace('_only','')}** vs "
                     f"**{b_key.replace('_only','')}**")
        lines.append("")
        lines.append("| 象限 | 数量 | 占比 |")
        lines.append("| :--- | ---: | ---: |")
        for k in ('both_correct', a_key, b_key, 'both_wrong'):
            lines.append(f"| {k} | {quad[k]} | {quad[k]/total*100:.2f}% |")
        lines.append("")
        lines.append(f"- `{b_key} = {quad[b_key]}`"
                     + ("，**存在静态失效而融合有效的样本，互补性证据成立**"
                        if quad[b_key] > 0 else "，互补性证据不足"))
        lines.append("")

    if sig:
        lines.append("## 配对 t 检验（5.1）")
        lines.append("")
        lines.append("| 对比 | 平均提升 | t | p | Cohen's d |")
        lines.append("| :--- | ---: | ---: | ---: | ---: |")
        for r in sig:
            lines.append(f"| {r['comparison']} | {r['mean_gain']*100:+.2f}pp | "
                         f"{r['t_statistic']:.4f} | {r['p_value']:.6f} | "
                         f"{r['cohens_d']:.4f} |")
        lines.append("")

    path = os.path.join(out_dir, 'summary.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\n📝 汇总报告已保存: {path}")


if __name__ == '__main__':
    main()
