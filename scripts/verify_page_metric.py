"""
验证页面级准确率指标是否可信。

跑法（云端）：
    python -u scripts/verify_page_metric.py \
        --ckpt outputs/checkpoints/arch_dcnn_best.pth \
        --metadata data/features/metadata.csv

三件事：
  1) K 曲线 —— 每页只用 K 个伪字符做聚合，看准确率怎么随 K 增长。
     如果 K=1 时明显高于 char_acc，说明指标有问题。
  2) 置换检验 —— 把页面→书写者的映射随机打乱，准确率必须掉回 ~0.1%。
     如果不掉，说明标签泄漏。
  3) 干净训练集上的 char_acc —— 判断到底是欠拟合还是记忆。
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import WPTTDataset, STATIC_FEATURE_VERSION
from src.data.dataloader_utils import make_loader
from src.evaluation.evaluator import _pad_to_96
from src.models.dual_stream import build_model


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--metadata', default='data/features/metadata.csv')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--max_pages', type=int, default=0, help='0=全部')
    p.add_argument('--cache_dir', default='data/cache',
                   help='切分缓存目录；传空字符串关闭缓存')
    p.add_argument('--skip_train', action='store_true',
                   help='跳过第 3 项（干净训练集 char_acc），可省很多时间')
    return p.parse_args()


@torch.no_grad()
def collect(model, dataset, device, batch_size, num_workers):
    loader = make_loader(dataset, batch_size, num_workers, shuffle=False,
                         drop_last=False)
    all_p, all_l = [], []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True))
        if isinstance(out, dict):
            out = out.get('logits', next(iter(out.values())))
        all_p.append(torch.softmax(out.float(), dim=1).cpu().numpy())
        all_l.append(np.asarray(y))
    return np.concatenate(all_p), np.concatenate(all_l)


def main():
    a = parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    arch = ck.get('arch', 'dcnn')
    # 训练器把 static_feature_version / mode 放在 'config' 子字典里，
    # 不在顶层（见 trainer_multi._save_single）。两处都兼容。
    cfg = ck.get('config', {}) or {}
    ver = ck.get('static_feature_version', cfg.get('static_feature_version'))
    mode = ck.get('mode', cfg.get('mode'))
    print(f"检查点: arch={arch}  mode={mode}  epoch={ck.get('epoch')}  "
          f"best_acc={ck.get('best_acc')}  特征版本={ver}")
    if mode not in (None, 'static'):
        raise SystemExit(f"该检查点是 {mode} 模式（非纯静态），本脚本只支持 static")
    if ver not in (None, STATIC_FEATURE_VERSION):
        print(f"⚠️  特征版本不一致（模型 {ver} vs 代码 {STATIC_FEATURE_VERSION}）")
        raise SystemExit("特征版本不匹配，评测结果无意义，终止")

    md = pd.read_csv(a.metadata, encoding='utf-8-sig')
    md['writer_id'] = md['writer_id'].astype(str)
    writers = sorted(md['writer_id'].unique())
    label_map = {w: i for i, w in enumerate(writers)}
    n_cls = len(writers)

    # 训练器保存的是 DualStreamWriterID（mode='static'），权重键名带
    # `static_branch.` 前缀；不能用裸骨干 build_static_backbone 去加载。
    # 这里用检查点里自带的 config 复原同一个对象。
    model_cfg = dict(cfg)
    model_cfg['mode'] = 'static'
    model_cfg['arch'] = arch
    model_cfg.setdefault('in_chans', 64)
    model_cfg.setdefault('feature_dim', 512)
    model_cfg.setdefault('pretrained', False)

    ds = WPTTDataset(a.metadata, split='Test', transform=_pad_to_96,
                     page_workers=0, segment_workers=0, apply_drop=False,
                     augment_points=False, cache_dir=(a.cache_dir or None))
    # 数据集内部的 label_map 是按 split 重排的；本项目里所有书写者都同时出现在
    # Train/Test，所以它与全局映射一致。断言一下，避免今后划分变化时静默错位。
    if ds.label_map != label_map:
        raise SystemExit("数据集内部 label_map 与全局映射不一致，请检查划分")

    model = build_model(model_cfg, n_cls, pretrained_path=None).to(device)
    sd = ck.get('model_state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"⚠️  权重不完全匹配：缺失 {len(missing)} 项，多余 {len(unexpected)} 项")
        print(f"   缺失示例: {missing[:5]}")
        print(f"   多余示例: {unexpected[:5]}")
        raise SystemExit("检查点与模型结构不匹配，终止")
    print("✅ 权重完全匹配")
    model.eval()

    probs, _ = collect(model, ds, device, a.batch_size, a.num_workers)
    print(f"收集到 {len(probs)} 个伪字符概率")

    pages = np.array(ds.page_names_for_samples)
    uniq = sorted(set(pages.tolist()))
    if a.max_pages:
        uniq = uniq[:a.max_pages]
    pid = {p: i for i, p in enumerate(uniq)}
    page_idx = np.array([pid.get(p, -1) for p in pages])
    keep = page_idx >= 0
    probs, page_idx = probs[keep], page_idx[keep]

    # 每页真实标签
    page_writer = {}
    for _, r in md[md['split'] == 'Test'].iterrows():
        page_writer[r['page']] = label_map[r['writer_id']]
    true_lab = np.array([page_writer[p] for p in uniq])

    # 每页样本下标
    buckets = [[] for _ in uniq]
    for i, pi in enumerate(page_idx):
        buckets[pi].append(i)
    counts = np.array([len(b) for b in buckets])
    print(f"页面数 {len(uniq)}，每页伪字符 中位 {int(np.median(counts))} "
          f"（范围 {counts.min()}~{counts.max()}）")

    def page_acc(K=None, labels=None, n_rep=5, seed=0):
        rng = np.random.default_rng(seed)
        labels = true_lab if labels is None else labels
        accs = []
        for _ in range(n_rep if K is not None else 1):
            hit = 0
            for pi, b in enumerate(buckets):
                idx = b if K is None or K >= len(b) else \
                    rng.choice(b, size=K, replace=False).tolist()
                m = probs[idx].mean(axis=0)
                if int(m.argmax()) == labels[pi]:
                    hit += 1
            accs.append(hit / len(uniq))
        return float(np.mean(accs))

    print("\n=== 1) K 曲线（每页只用 K 个伪字符聚合）===")
    print("   %-10s %-12s" % ('K', 'page_acc'))
    for K in (1, 2, 5, 10, 20, 50, 100, None):
        print("   %-10s %-12s" % ('全部' if K is None else str(K),
                                  '%.2f%%' % (page_acc(K) * 100)))

    print("\n=== 2) 置换检验（页面标签随机打乱，应掉回 ~%.2f%%）===" %
          (100.0 / n_cls))
    rng = np.random.default_rng(123)
    for t in range(3):
        perm = rng.permutation(true_lab)
        print("   第%d次  打乱后 page_acc = %.2f%%" %
              (t + 1, page_acc(None, labels=perm) * 100))
    print("   原始      page_acc = %.2f%%" % (page_acc(None) * 100))

    print("\n=== 3) 干净训练集上的 char_acc（判断记忆 vs 欠拟合）===")
    if a.skip_train:
        print("   已按 --skip_train 跳过")
        return
    tr = WPTTDataset(a.metadata, split='Train', transform=_pad_to_96,
                     page_workers=0, segment_workers=0, apply_drop=False,
                     augment_points=False, cache_dir=(a.cache_dir or None))
    n_eval = min(60000, len(tr))
    sub = torch.utils.data.Subset(tr, np.random.default_rng(0)
                                  .choice(len(tr), n_eval, replace=False).tolist())
    p_tr, y_tr = collect(model, sub, device, a.batch_size, a.num_workers)
    tr_char = float((p_tr.argmax(1) == y_tr).mean())
    print("   训练集（干净、无增强）char_acc = %.2f%%   （%d 样本）" %
          (tr_char * 100, n_eval))
    te_char = float((probs.argmax(1) ==
                     np.array([true_lab[page_idx[i]] for i in range(len(probs))])).mean())
    print("   测试集                    char_acc = %.2f%%" % (te_char * 100))
    print("\n   判读：若训练集 char_acc 远超测试集 -> 记忆；"
          "若两者接近且都低 -> 欠拟合但指标健康。")


if __name__ == '__main__':
    main()
