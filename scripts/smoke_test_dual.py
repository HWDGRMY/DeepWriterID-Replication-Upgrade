"""
双路融合全套件的本地冒烟测试。

**不训练到收敛**，只验证「代码能不能跑通、接口对不对、静态基线是否等价」。
在有 GPU 的机器上约 2-5 分钟跑完（不含最后的端到端小样本训练环节）。

用法:
    python scripts/smoke_test_dual.py                # 全部检查
    python scripts/smoke_test_dual.py --skip_e2e     # 跳过小样本端到端训练（最快）
    python scripts/smoke_test_dual.py --ckpt outputs/checkpoints/convnext_best_9617.pth

覆盖:
     1. 全部模块可导入
     2. DynamicBranch：前向、形状、参数量、池化/位置编码变体
     3. ★ 静态数值等价：mode='static' 与 ConvNeXtBackbone 逐元素完全一致
        （这是消融实验 A「= 96.17% 基线」成立的前提）
     4. 三种模式（static/dynamic/dual）前向 + 反向
     5. 冻结/解冻：梯度是否真的被切断、冻结时是否强制 eval
     6. ★ load_static_weights：能否吃下旧的 convnext_best 检查点
     7. 三阶段优化器/调度器构造 + 分组学习率正确性
     8. WPTTDataset 三元组端到端（metadata_small.csv）
     9. 评估器：批次解包、逐样本特征导出
    10. ★ 训练器端到端：小样本跑 1 个 epoch，验证完整训练回路
"""

from __future__ import annotations

import os
import sys
import argparse
import traceback

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass


class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
        return ok

    def summary(self):
        print("\n" + "=" * 74)
        print(f"结果: {self.passed} 项通过, {self.failed} 项失败")
        print("=" * 74)
        return self.failed == 0


# ---------------------------------------------------------------------------

def make_subset_metadata(src_metadata, out_path, n_writers=3,
                         train_pages_per_writer=2, test_pages_per_writer=1):
    """
    从完整 metadata 里抽一个极小子集，让端到端检查在 CPU 上也能分钟级跑完。

    注意：真实训练请直接用完整的 metadata.csv，不要用这个子集。
    """
    import pandas as pd

    df = pd.read_csv(src_metadata, encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)
    writers = sorted(df['writer_id'].unique())[:n_writers]

    parts = []
    for split, per in (('Train', train_pages_per_writer),
                       ('Test', test_pages_per_writer)):
        sub = df[(df['split'] == split) & (df['writer_id'].isin(writers))]
        parts.append(sub.groupby('writer_id', group_keys=False).head(per))
    sub = pd.concat(parts, ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sub.to_csv(out_path, index=False, encoding='utf-8-sig')
    n_tr = int((sub['split'] == 'Train').sum())
    n_te = int((sub['split'] == 'Test').sum())
    print(f"   已生成子集 metadata: {out_path} "
          f"(writers={len(writers)}, Train={n_tr} 页, Test={n_te} 页)")
    return out_path


def check_static_variants(rep, n_pages=3):
    """
    ★ 静态特征流水线的回归护栏。

    A0（会抹平抬笔分隔符的旧实现）已从代码里删除，这里改为验证修正版
    必须满足的四条不变量 —— 任何一条被破坏都会导致特征静默退化：

      1. 抬笔分隔符必须被保留（`pseudo_segment` 内部插 -1）；
      2. 对伪字符再切分一次必须**仍然只有一块**。早期版本用 `chars_drop[0]`
         静默取第一块，实测在 A0 下 Train 24.9% / Test 22.3% 的样本被截断，
         平均丢弃 21.9% 的墨迹，而样本数、张量形状全都看不出来；
      3. 特征图必须是 (54, 54, 64) = 1 位图 + 63 签名；
      4. 63 个签名通道必须有真实的空间结构（A0 下它们退化成常数底图，
         空间结构只剩位图的 3.8%）。
    """
    print("\n[11] ★ 静态特征流水线不变量")
    import glob
    from src.preprocessing.segmentation import pseudo_segment, estimate_avg_char_height
    from src.data.loader import load_wptt_page
    from src.preprocessing.corner import detect_corners
    from src.data.dataset import WPTTDataset

    files = sorted(glob.glob(os.path.join('data', 'raw', 'WPTT2.0-Train', '*.wptt')))
    if not files:
        print("  [SKIP] 未找到 .wptt 文件")
        return
    files = files[:n_pages]

    n_chars = 0
    n_sep = 0
    n_multi = 0
    hw = []
    for fp in files:
        pts = load_wptt_page(fp)
        corners = detect_corners(pts, k=2, threshold=3)
        # 与 WPTTDataset.__init__ 一致：先量平均字高，再切
        avg_h = estimate_avg_char_height(pts, corners)
        chars = pseudo_segment(pts, corners, avg_height=avg_h)
        n_chars += len(chars)
        for ch in chars:
            if ch.ndim == 1:
                ch = ch.reshape(-1, 2)
            n_sep += int((ch[:, 0] == -1.0).sum())
            d = ch[ch[:, 0] != -1.0]
            if len(d) >= 3:
                h = float(d[:, 1].max() - d[:, 1].min())
                w = float(d[:, 0].max() - d[:, 0].min())
                if w > 1e-6:
                    hw.append(h / w)
            # 与 __getitem__ 一致：**沿用该页**的阈值重切。
            # 不传阈值会按单字重新估计（得到很小的值），必然把字切开——
            # 那是测试写法错误，不是流水线问题。
            again = pseudo_segment(ch, detect_corners(ch, k=2, threshold=3),
                                   avg_height=avg_h)
            if len(again) != 1:
                n_multi += 1

    rep.check("产出了伪字符", n_chars > 0, f"{n_chars} 个")
    rep.check("★ 抬笔分隔符被保留（-1 行数 > 0）", n_sep > 0,
              f"{n_sep} 个分隔符")
    rep.check("★ 伪字符二次切分几乎恒为 1 块（沿用页级阈值）",
              n_multi / max(n_chars, 1) < 0.01,
              f"{n_multi}/{n_chars} 个被切开")
    # 论文要求 height-to-width ratio ≈ 1（[1, 1.3] 亦可）
    med_hw = float(np.median(hw)) if hw else 0.0
    rep.check("★ 伪字符高宽比接近 1（论文 §2.2）", 0.85 <= med_hw <= 1.3,
              f"h/w 中位 {med_hw:.2f}")

    pts = load_wptt_page(files[0])
    for ch in pseudo_segment(pts, detect_corners(pts, k=2, threshold=3)):
        if ch.ndim == 1:
            ch = ch.reshape(-1, 2)
        d = ch[ch[:, 0] != -1.0]
        if len(d) < 3:
            continue
        feat = WPTTDataset._build_static_feature(ch, d)
        rep.check("特征图形状为 (54, 54, 64)", feat.shape == (54, 54, 64),
                  str(feat.shape))
        sp_sig = float(feat[:, :, 1:].std(axis=(0, 1)).mean())
        sp_bmp = float(feat[:, :, 0].std())
        rep.check("★ 签名通道具备空间结构（不是常数底图）",
                  sp_sig > 1e-3, f"通道平均空间标准差 {sp_sig:.5f}")
        rep.check("位图通道与签名通道的量级同阶（不是 3.8% 那种退化）",
                  sp_sig > 0.1 * max(sp_bmp, 1e-9),
                  f"签名 {sp_sig:.4f} vs 位图 {sp_bmp:.4f}")
        break


def check_arch_switch(rep):
    """
    ★ 静态骨干网切换（ConvNeXt / 原论文 DCNN）。

    两个骨干网必须满足同一份输入输出契约，否则「架构对比」就混进了
    数据管道差异。同时校验 DCNN 的权重键名与原复刻仓库一致，
    以便对方的 dcnn_best.pth 能直接加载。
    """
    print("\n[12] ★ 静态骨干网切换：ConvNeXt / 原论文 DCNN")
    from src.models.dual_stream import (DualStreamWriterID, build_static_backbone,
                                        STATIC_ARCHS)
    from src.models.dcnn_backbone import DCNNBackbone

    rep.check("STATIC_ARCHS 含 convnext 与 dcnn",
              set(STATIC_ARCHS) >= {'convnext', 'dcnn'}, f"{STATIC_ARCHS}")

    torch.manual_seed(0)
    x = torch.randn(2, 64, 96, 96)
    shapes_ok = True
    n_params = {}
    for arch in ('convnext', 'dcnn'):
        m = DualStreamWriterID(num_classes=1019, mode='static',
                               static_arch=arch, pretrained=False,
                               freeze_static=False)
        m.train()
        logits = m(x)
        feats = m(x, return_features=True)
        if tuple(logits.shape) != (2, 1019) or tuple(feats.shape) != (2, 512):
            shapes_ok = False
            rep.check(f"{arch} 输出契约", False,
                      f"logits={tuple(logits.shape)} feats={tuple(feats.shape)}")
        logits.sum().backward()
        n_params[arch] = m.num_total_params()

    rep.check("两个骨干网输出契约一致 (B,1019) / (B,512)", shapes_ok)
    rep.check("参数量符合预期（ConvNeXt≈5.6M / DCNN≈2.6M）",
              5.0e6 < n_params.get('convnext', 0) < 6.5e6
              and 2.0e6 < n_params.get('dcnn', 0) < 3.2e6,
              f"cn={n_params.get('convnext',0)/1e6:.3f}M "
              f"dcnn={n_params.get('dcnn',0)/1e6:.3f}M")

    # ---- DCNN 结构是否符合论文 ----
    d = DCNNBackbone(num_classes=1019)
    flat = DCNNBackbone._infer_flat_dim(64)
    rep.check("DCNN 展平维度 = 1600（论文 400×2×2）", flat == 1600, f"{flat}")
    conv_ch = [d.conv1[0].out_channels, d.conv2[0].out_channels,
               d.conv3[0].out_channels, d.conv4[0].out_channels,
               d.conv5[0].out_channels]
    rep.check("DCNN 卷积通道 80/160/240/320/400（论文：每次池化 +80）",
              conv_ch == [80, 160, 240, 320, 400], f"{conv_ch}")
    ks = [d.conv1[0].kernel_size[0], d.conv2[0].kernel_size[0]]
    rep.check("DCNN 首层 C3、其余 C2", ks == [3, 2], f"{ks}")
    rep.check("DCNN FC 为 480 → 512（论文）",
              d.fc1[0].out_features == 480 and d.fc2[0].out_features == 512)
    rep.check("DCNN 无归一化层（原架构如此）",
              not any(isinstance(mm, (nn.BatchNorm2d, nn.LayerNorm))
                      for mm in d.modules()))

    # ---- 键名与复刻仓库兼容 ----
    keys = set(d.state_dict().keys())
    expect = {'conv1.0.weight', 'conv1.0.bias', 'conv5.0.weight',
              'fc1.0.weight', 'fc2.0.weight', 'output.1.weight',
              'output.1.bias'}
    rep.check("★ DCNN 权重键名与复刻仓库一致（其 dcnn_best.pth 可直接加载）",
              expect <= keys, f"缺失 {sorted(expect - keys)}")

    try:
        wrapper = build_static_backbone('dcnn', num_classes=1019)
        miss = wrapper.load_state_dict(d.state_dict(), strict=False)
        rep.check("裸 DCNN 权重可无缝加载进框架",
                  not [k for k in miss.missing_keys if 'classifier' not in k]
                  and not miss.unexpected_keys,
                  f"missing={len(miss.missing_keys)} "
                  f"unexpected={len(miss.unexpected_keys)}")
    except Exception as exc:                          # noqa: BLE001
        rep.check("裸 DCNN 权重可无缝加载进框架", False, str(exc))

    try:
        build_static_backbone('vgg', num_classes=10)
        rep.check("未知架构应报错", False, "未报错")
    except ValueError:
        rep.check("未知架构应报错", True)


def check_trainer_multi(rep, metadata, seq_len, work_dir='_smoke_out'):
    """★ 共享管道的多模型训练器：一份数据同时喂两个模型。"""
    print("\n[13] ★ 共享管道多模型训练（ConvNeXt + DCNN 同时）")
    from src.training.trainer_multi import train_multi

    cfg = {
        'runs': [
            {'name': 'cn', 'arch': 'convnext', 'mode': 'static',
             'prefix': 'smoke_cn', 'lr': 1e-3, 'optimizer': 'adamw',
             'weight_decay': 0.05, 'warmup_epochs': 0, 'eval_frequency': 1},
            {'name': 'dc', 'arch': 'dcnn', 'mode': 'static',
             'prefix': 'smoke_dc', 'lr': 1e-3, 'optimizer': 'adam',
             'weight_decay': 0.0, 'warmup_epochs': 0, 'eval_frequency': 1},
        ],
        'model': {'feature_dim': 512, 'in_chans': 64, 'pretrained': False,
                  'drop_path': 0.0, 'dropout_head': 0.1,
                  'dropout_classifier': 0.0, 'dcnn_dropout': 0.0,
                  'dynamic_dim': 256, 'seq_len': seq_len, 'dynamic_in_dim': 2,
                  'd_model': 64, 'nhead': 4, 'num_layers': 2,
                  'dim_feedforward': 128, 'dynamic_dropout': 0.0,
                  'pooling': 'attention', 'pos_encoding': 'sinusoidal',
                  'fusion_hidden': 128, 'fusion_dropout': 0.1},
        'batch_size': 8, 'num_workers': 0, 'page_workers': 0,
        'segment_workers': 0, 'epochs': 1, 'grad_clip': 5.0,
        'seq_len': seq_len, 'dynamic_speed_clip': 10.0,
        'apply_drop': True, 'resume': False,
        'metadata_file': metadata,
        'checkpoint': {'save_dir': os.path.join(work_dir, 'mckpt'),
                       'group': 'smoke_multi'},
        'logging': {'log_dir': os.path.join(work_dir, 'mlogs'),
                    'log_file': 'smoke_multi.csv'},
    }
    try:
        res = train_multi(cfg)
    except Exception as exc:                          # noqa: BLE001
        rep.check("多模型训练回路", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    rep.check("两个模型都跑通", len(res['runs']) == 2, f"{list(res['runs'])}")
    for name in ('cn', 'dc'):
        info = res['runs'].get(name)
        rep.check(f"{name} 检查点已落盘",
                  info is not None and os.path.exists(info['best_path']),
                  info['best_path'] if info else 'N/A')

    log_csv = res['log_path']
    if os.path.exists(log_csv):
        with open(log_csv, encoding='utf-8-sig') as f:
            header = f.readline().strip().split(',')
        rep.check("日志表头含两个模型各自的 4 列",
                  len(header) == 2 + 2 * 4
                  and 'cn_loss' in header and 'dc_page_acc' in header,
                  f"{header}")
    else:
        rep.check("日志文件存在", False, log_csv)

    try:
        ck = torch.load(res['combined_path'], map_location='cpu',
                        weights_only=False)
        rep.check("组合断点含两个模型 + 优化器/调度器状态",
                  set(ck['run_names']) == {'cn', 'dc'}
                  and set(ck['optimizers']) == {'cn', 'dc'}
                  and set(ck['schedulers']) == {'cn', 'dc'})
    except Exception as exc:                          # noqa: BLE001
        rep.check("组合断点可读", False, str(exc))


def check_dcnn_fidelity(rep):
    """
    ★ DCNN 与原仓库的**逐项保真**回归护栏。

    起因：早期版本把 ConvNeXt 的 `dropout_classifier=0.4` 也传给了 DCNN，
    而原 DCNN 的分类头自带 Dropout(0.2)，于是它变成
        Dropout(0.4) → Dropout(0.2) → Linear
    有效 dropout 从 0.2 涨到 0.52，分类头被过度正则化 —— 而**参数量不变**，
    光看参数量根本发现不了。本函数逐项比对，防止再犯。
    """
    print("\n[14] ★ DCNN 精度保真（逐项对比原仓库实现）")
    from src.models.dual_stream import build_static_backbone

    # ---- 按原仓库 dcnn.py 原样搭一个参照实现 ----
    class OrigDCNN(nn.Module):
        def __init__(self, num_classes=1019):
            super().__init__()
            self.conv1 = nn.Sequential(nn.Conv2d(64, 80, 3), nn.ReLU(True),
                                       nn.MaxPool2d(2, 2))
            self.conv2 = nn.Sequential(nn.Conv2d(80, 160, 2), nn.ReLU(True),
                                       nn.MaxPool2d(2, 2))
            self.conv3 = nn.Sequential(nn.Conv2d(160, 240, 2), nn.ReLU(True),
                                       nn.MaxPool2d(2, 2))
            self.conv4 = nn.Sequential(nn.Conv2d(240, 320, 2), nn.ReLU(True),
                                       nn.MaxPool2d(2, 2))
            self.conv5 = nn.Sequential(nn.Conv2d(320, 400, 2), nn.ReLU(True),
                                       nn.MaxPool2d(2, 2))
            self.fc1 = nn.Sequential(nn.Linear(1600, 480), nn.ReLU(True),
                                     nn.Dropout(0.2))
            self.fc2 = nn.Sequential(nn.Linear(480, 512), nn.ReLU(True),
                                     nn.Dropout(0.2))
            self.output = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    orig = OrigDCNN(1019)
    mine = build_static_backbone('dcnn', num_classes=1019, in_chans=64,
                                 feature_dim=512, dcnn_dropout=0.2)

    n_orig = sum(p.numel() for p in orig.parameters())
    n_mine = sum(p.numel() for p in mine.parameters())
    rep.check("参数量与原仓库一致", n_orig == n_mine == 2608779,
              f"原={n_orig} 我={n_mine}")

    # 参数量一样不代表前向一样：逐项比 Dropout
    def drops(m):
        return sorted((n, mod.p) for n, mod in m.named_modules()
                      if isinstance(mod, nn.Dropout))

    d_orig = drops(orig)
    d_mine = [(n, p) for n, p in drops(mine) if p > 0]
    rep.check("★ Dropout 集合与原仓库完全一致（不能多、p 不能变）",
              d_orig == d_mine, f"原={d_orig} 我={d_mine}")

    # 关键判据：分类层（output.1）正前方只能有 output.0 这一个 Dropout(0.2)。
    # 早期 bug 会在它前面多插一个 classifier_dropout(0.4)，这里直接抓。
    mods = dict(mine.named_modules())
    rep.check("分类层正前方是单个 Dropout(0.2)（原 output.0）",
              isinstance(mods.get('output.0'), nn.Dropout)
              and abs(mods['output.0'].p - 0.2) < 1e-9,
              f"p={getattr(mods.get('output.0'), 'p', None)}")
    extra = mods.get('classifier_dropout')
    rep.check("不存在多余的分类头 dropout（classifier_dropout 必须缺失或 p=0）",
              extra is None or (isinstance(extra, nn.Dropout) and extra.p == 0.0),
              f"classifier_dropout={None if extra is None else extra.p}")
    rep.check("DCNN 的 Dropout 总数 = 3（原仓库: fc1.2 / fc2.2 / output.0）",
              len([1 for _, x in mine.named_modules()
                   if isinstance(x, nn.Dropout) and x.p > 0]) == 3)

    # ConvNeXt 侧必须保持 dropout_classifier=0.4（原 trainer.py 用法）
    cx = build_static_backbone('convnext', num_classes=1019, in_chans=64,
                               feature_dim=512, dropout_classifier=0.4)
    cx_p = dict(drops(cx))
    rep.check("ConvNeXt 的 classifier.0 dropout 仍为 0.4",
              cx_p.get('classifier.0') == 0.4, f"{cx_p.get('classifier.0')}")
    rep.check("ConvNeXt 的 feature_head dropout 仍为 0.3",
              cx_p.get('feature_head.2') == 0.3, f"{cx_p.get('feature_head.2')}")

    # 键名必须与原仓库一致（旧 dcnn_best.pth 可直接加载）
    need = {'conv1.0.weight', 'fc1.0.weight', 'fc2.0.weight',
            'output.1.weight', 'output.1.bias'}
    rep.check("权重键名与原仓库一致", need <= set(mine.state_dict().keys()))


def check_imports(rep):
    print("\n[1] 模块导入")
    mods = [
        "src.features.dynamic",
        "src.models.dynamic_branch",
        "src.models.dual_stream",
        "src.models.backbone",
        "src.data.dataset",
        "src.evaluation.evaluator",
        "src.training.trainer_dual",
    ]
    ok = True
    for m in mods:
        try:
            __import__(m)
        except Exception as exc:                      # noqa: BLE001
            ok = False
            rep.check(m, False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
    rep.check("全部核心模块导入成功", ok)
    return ok


def check_dynamic_branch(rep):
    print("\n[2] DynamicBranch")
    from src.models.dynamic_branch import DynamicBranch

    torch.manual_seed(0)
    x = torch.randn(4, 200, 2)

    db = DynamicBranch(in_dim=2, d_model=64, nhead=4, num_layers=4,
                       dim_feedforward=256, out_dim=256, seq_len=200)
    y = db(x)
    rep.check("输出形状 (B, out_dim)", tuple(y.shape) == (4, 256), f"{tuple(y.shape)}")

    n_param = sum(p.numel() for p in db.parameters())
    rep.check("参数量在合理范围", 0.05e6 < n_param < 5e6, f"{n_param/1e6:.3f}M")

    n_pool = sum(p.numel() for p in db.attn_pool.parameters())
    rep.check("注意力池化仅增加 65 个参数（d_model=64）", n_pool == 65,
              f"{n_pool} 个")

    for pooling in ("attention", "mean", "max"):
        try:
            m = DynamicBranch(pooling=pooling)
            o = m(x)
            assert o.shape == (4, 256)
        except Exception as exc:                      # noqa: BLE001
            rep.check(f"pooling={pooling}", False, str(exc))
            return
    rep.check("三种池化方式均可用", True)

    for pe in ("sinusoidal", "learnable", "none"):
        try:
            m = DynamicBranch(pos_encoding=pe)
            assert m(x).shape == (4, 256)
        except Exception as exc:                      # noqa: BLE001
            rep.check(f"pos_encoding={pe}", False, str(exc))
            return
    rep.check("三种位置编码均可用", True)

    # 反向传播
    y.sum().backward()
    grads_ok = all(p.grad is not None for p in db.parameters() if p.requires_grad)
    rep.check("反向传播后所有参数都有梯度", grads_ok)

    # 非法配置应报错
    try:
        DynamicBranch(d_model=63, nhead=4)
        rep.check("d_model 不被 nhead 整除时应报错", False, "未报错")
    except ValueError:
        rep.check("d_model 不被 nhead 整除时应报错", True)


def _maxdiff(a, b):
    """最大绝对差（detach 以避免 requires_grad 张量转标量的告警）。"""
    return float((a - b).abs().max().detach())


def check_static_equivalence(rep):
    """mode='static' 必须与 ConvNeXtBackbone 数值完全一致。"""
    print("\n[3] ★ 静态数值等价性（消融实验 A 成立的前提）")
    from src.models.backbone import ConvNeXtBackbone
    from src.models.dual_stream import DualStreamWriterID

    torch.manual_seed(0)
    base = ConvNeXtBackbone(num_classes=10, in_chans=64, feature_dim=512,
                            pretrained=False)
    dual = DualStreamWriterID(num_classes=10, mode='static', pretrained=False,
                              freeze_static=False)
    missing = dual.static_branch.load_state_dict(base.state_dict(), strict=True)
    rep.check("权重可无缝搬移", not missing.missing_keys and not missing.unexpected_keys)

    base.eval()
    dual.eval()
    x = torch.randn(2, 64, 96, 96)
    with torch.no_grad():
        a = base(x)
        b = dual(x)
    rep.check("logits 逐元素完全一致", torch.equal(a, b),
              f"最大差={_maxdiff(a, b):.3e}")

    with torch.no_grad():
        fa = base(x, return_features=True)
        fb = dual(x, return_features=True)
    rep.check("512 维静态特征逐元素一致", torch.equal(fa, fb),
              f"最大差={_maxdiff(fa, fb):.3e}")

def check_three_modes(rep):
    print("\n[4] 三种模式前向 + 反向")
    from src.models.dual_stream import DualStreamWriterID

    torch.manual_seed(0)
    s = torch.randn(2, 64, 96, 96)
    d = torch.randn(2, 200, 2)

    cases = [('static', s, None, (2, 10)), ('dynamic', None, d, (2, 10)),
             ('dual', s, d, (2, 10))]
    for mode, si, di, expect in cases:
        try:
            m = DualStreamWriterID(num_classes=10, mode=mode, pretrained=False,
                                   freeze_static=False)
            m.train()
            out = m(si, di)
            assert tuple(out.shape) == expect, f"shape={tuple(out.shape)}"
            out.sum().backward()
            n_grad = sum(1 for p in m.parameters()
                         if p.requires_grad and p.grad is not None)
            assert n_grad > 0, "没有任何参数获得梯度"
            rep.check(f"mode={mode} 前向/反向正常", True,
                      f"参数量={m.num_total_params()/1e6:.2f}M")
        except Exception as exc:                      # noqa: BLE001
            rep.check(f"mode={mode}", False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    # 纯静态模式绝不能被冻结成空转
    m = DualStreamWriterID(num_classes=10, mode='static', pretrained=False,
                           freeze_static=True)
    rep.check("mode='static' 忽略冻结请求（保持可训练）",
              not m.static_frozen and m.num_trainable_params() > 0,
              f"可训练={m.num_trainable_params()/1e6:.2f}M")

    # 特征接口
    m = DualStreamWriterID(num_classes=10, mode='dual', pretrained=False,
                           freeze_static=False)
    fs, fd = m.forward_features(s, d)
    rep.check("forward_features 返回 (512, 256)",
              tuple(fs.shape) == (2, 512) and tuple(fd.shape) == (2, 256),
              f"{tuple(fs.shape)} / {tuple(fd.shape)}")
    fused = m(s, d, return_features=True)
    rep.check("dual 融合特征为 768 维", tuple(fused.shape) == (2, 768),
              f"{tuple(fused.shape)}")


def check_freeze(rep):
    print("\n[5] 冻结 / 解冻语义")
    from src.models.dual_stream import DualStreamWriterID

    torch.manual_seed(0)
    s = torch.randn(2, 64, 96, 96)
    d = torch.randn(2, 200, 2)

    m = DualStreamWriterID(num_classes=10, mode='dual', pretrained=False,
                           freeze_static=True)
    rep.check("冻结后 static_frozen=True", m.static_frozen)
    rep.check("冻结时静态分支参数 requires_grad=False",
              not any(p.requires_grad for p in m.static_branch.parameters()))

    # 冻结时应强制 eval（消除 drop_path / dropout 随机性）
    m.train()
    rep.check("冻结时调用 train() 后静态分支仍为 eval",
              not m.static_branch.training, f"training={m.static_branch.training}")
    rep.check("冻结时动态分支仍为 train",
              m.dynamic_branch.training, f"training={m.dynamic_branch.training}")

    # 冻结时静态特征必须确定性
    m.eval()
    with torch.no_grad():
        f1 = m.static_branch(s, return_features=True)
        f2 = m.static_branch(s, return_features=True)
    rep.check("冻结时静态特征确定（无随机增强）", torch.equal(f1, f2))

    # 冻结时静态分支不应拿到梯度
    m.train()
    out = m(s, d)
    out.sum().backward()
    got_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.static_branch.parameters())
    rep.check("冻结时静态分支梯度为零（未参与更新）", not got_grad)

    # 解冻后应重新可训练
    m.set_static_frozen(False)
    rep.check("解冻后 requires_grad 恢复",
              all(p.requires_grad for p in m.static_branch.parameters()))
    m.train()
    m.zero_grad(set_to_none=True)
    out = m(s, d)
    out.sum().backward()
    got_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.static_branch.parameters())
    rep.check("解冻后静态分支可拿到梯度", got_grad)


def check_load_static_weights(rep, ckpt_path):
    print("\n[6] ★ 加载旧 ConvNeXtBackbone 检查点")
    from src.models.dual_stream import DualStreamWriterID

    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"  [SKIP] 未找到检查点: {ckpt_path}")
        return

    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    keys = list((raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw).keys())
    print(f"       检查点含 {len(keys)} 个张量，示例: {keys[:3]}")
    if isinstance(raw, dict) and 'epoch' in raw:
        print(f"       来源 epoch={raw.get('epoch')}, best_acc={raw.get('best_acc')}")

    m = DualStreamWriterID(num_classes=1019, mode='dual', pretrained=False,
                           freeze_static=False)
    info = m.load_static_weights(ckpt_path, strict=False, verbose=False)

    rep.check("成功加载静态分支权重", info['loaded'] > 0,
              f"loaded={info['loaded']}")
    # backbone + feature_head 为核心；classifier 形状匹配时也应载入
    core_missing = [k for k in info['missing'] if '.classifier.' not in k]
    rep.check("骨干网 + 特征头无缺失", not core_missing,
              f"缺失={core_missing[:5]}")
    rep.check("无形状不匹配", not info['shape_mismatch'],
              f"{info['shape_mismatch'][:3]}")

    # 加载后的静态分支应当是可用的（前向不报错、无 NaN）
    m.eval()
    with torch.no_grad():
        f = m.static_branch(torch.randn(1, 64, 96, 96), return_features=True)
    rep.check("加载后静态分支前向输出有限", bool(torch.isfinite(f).all()))

    # 缺文件时应有明确报错
    try:
        m.load_static_weights('definitely_missing_file.pth')
        rep.check("缺失文件时应报 FileNotFoundError", False, "未报错")
    except FileNotFoundError:
        rep.check("缺失文件时应报 FileNotFoundError", True)


def check_optimizer_stages(rep):
    print("\n[7] 三阶段优化器 / 调度器")
    from src.models.dual_stream import DualStreamWriterID
    from src.training.trainer_dual import build_optimizer, build_scheduler

    torch.manual_seed(0)
    m = DualStreamWriterID(num_classes=10, mode='dual', pretrained=False,
                           freeze_static=True)

    stages = [
        {'name': 's1', 'lr': 1e-3, 'freeze_static': True, 'static_lr_scale': 0.0},
        {'name': 's2', 'lr': 5e-4, 'freeze_static': True, 'static_lr_scale': 0.0},
        {'name': 's3', 'lr': 1e-4, 'freeze_static': False, 'static_lr_scale': 0.1},
    ]

    for st in stages:
        opt, lr = build_optimizer(m, st, weight_decay=0.05)
        names = [g.get('name') for g in opt.param_groups]
        if st['name'] in ('s1', 's2'):
            ok = names == ['dynamic_fusion']
        else:
            ok = set(names) == {'static', 'dynamic_fusion'}
        rep.check(f"{st['name']} 参数分组正确", ok, f"groups={names}")

        if st['name'] == 's3':
            g = {x.get('name'): x['lr'] for x in opt.param_groups}
            ratio = g['static'] / g['dynamic_fusion']
            rep.check("阶段3 静态分支学习率为 0.1 倍",
                      abs(ratio - 0.1) < 1e-9, f"ratio={ratio:.4f}")

        sch = build_scheduler(opt, epochs=3, steps_per_epoch=10,
                              warmup_epochs=1, base_lr=lr)
        before = [g['lr'] for g in opt.param_groups]
        # 必须先 optimizer.step() 再 scheduler.step()，否则 PyTorch 会告警
        for p in m.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
        opt.step()
        sch.step()
        after = [g['lr'] for g in opt.param_groups]
        rep.check(f"{st['name']} 调度器可 step", len(after) == len(before),
                  f"lr {before[0]:.2e} → {after[0]:.2e}")

    # 重建优化器后静态分支的 requires_grad 应随阶段切换
    build_optimizer(m, stages[0], 0.05)
    rep.check("切回阶段1 后静态分支重新冻结",
              not any(p.requires_grad for p in m.static_branch.parameters()))
    build_optimizer(m, stages[2], 0.05)
    rep.check("切到阶段3 后静态分支解冻",
              all(p.requires_grad for p in m.static_branch.parameters()))


def check_dataset(rep, metadata, seq_len, no_mp):
    print(f"\n[8] WPTTDataset 三元组端到端（{metadata}）")
    from src.data.dataset import WPTTDataset

    pw = 0 if no_mp else 2
    try:
        ds = WPTTDataset(metadata, split='Test', transform=None,
                         page_workers=pw, segment_workers=pw,
                         apply_drop=True, return_dynamic=True,
                         seq_len=seq_len)
    except Exception as exc:                          # noqa: BLE001
        rep.check("构造数据集", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    rep.check("数据集非空", len(ds) > 0, f"{len(ds)} 个样本")

    try:
        item = ds[0]
    except Exception as exc:                          # noqa: BLE001
        rep.check("取第 0 个样本", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    rep.check("返回三元组", len(item) == 3, f"元素个数={len(item)}")
    static_t, dyn_t, label = item
    rep.check("静态张量形状 (64, 54, 54)",
              tuple(static_t.shape) == (64, 54, 54), f"{tuple(static_t.shape)}")
    rep.check("动态张量形状 (seq_len, 2)",
              tuple(dyn_t.shape) == (seq_len, 2), f"{tuple(dyn_t.shape)}")
    rep.check("动态张量有限", bool(torch.isfinite(dyn_t).all()))
    rep.check("动态速度通道非负", bool((dyn_t[:, 0] >= 0).all()),
              f"min={float(dyn_t[:, 0].min()):.4f}")
    rep.check("动态曲率通道落在 [0, π]",
              float(dyn_t[:, 1].min()) >= 0 and float(dyn_t[:, 1].max()) <= np.pi + 1e-5,
              f"[{float(dyn_t[:, 1].min()):.4f}, {float(dyn_t[:, 1].max()):.4f}]")
    rep.check("标签为非负整数", isinstance(label, int) and label >= 0,
              f"label={label}")

    # 向后兼容：return_dynamic=False 必须仍是二元组
    ds2 = WPTTDataset(metadata, split='Test', transform=None,
                      page_workers=pw, segment_workers=pw,
                      apply_drop=False, return_dynamic=False)
    item2 = ds2[0]
    rep.check("return_dynamic=False 时仍返回二元组（向后兼容）",
              len(item2) == 2, f"元素个数={len(item2)}")

    # 批量化
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    rep.check("DataLoader 可正常拼接三元组",
              len(batch) == 3 and tuple(batch[0].shape) == (4, 64, 54, 54)
              and tuple(batch[1].shape) == (4, seq_len, 2),
              f"{tuple(batch[0].shape)} / {tuple(batch[1].shape)}")


def check_evaluator(rep, metadata, seq_len, no_mp):
    print("\n[9] 评估器：批次解包 + 逐样本特征导出")
    from src.evaluation.evaluator import _unpack_batch
    from src.models.dual_stream import DualStreamWriterID

    a, b, c = _unpack_batch((torch.zeros(1), torch.zeros(1), torch.zeros(1)))
    rep.check("_unpack_batch 处理三元组",
              a is not None and b is not None and c is not None)
    a, b, c = _unpack_batch((torch.zeros(1), torch.zeros(1)))
    rep.check("_unpack_batch 处理二元组（动态位置为 None）",
              a is not None and b is None and c is not None)

    import pandas as pd
    df = pd.read_csv(metadata, encoding='utf-8-sig')
    df['writer_id'] = df['writer_id'].astype(str)
    writers = sorted(df[df['split'] == 'Test']['writer_id'].unique())
    label_map = {w: i for i, w in enumerate(writers)}

    device = torch.device('cpu')
    model = DualStreamWriterID(num_classes=len(writers), mode='dual',
                               pretrained=False, freeze_static=False).to(device)

    from src.evaluation.evaluator import evaluate_page_level, extract_test_features
    pw = 0 if no_mp else 2
    try:
        acc = evaluate_page_level(model, metadata, label_map, device,
                                  batch_size=16, num_workers=0,
                                  page_workers=pw, segment_workers=pw,
                                  apply_drop=False, return_dynamic=True,
                                  seq_len=seq_len)
        rep.check("evaluate_page_level 跑通", 0.0 <= acc <= 1.0, f"acc={acc:.4f}")
    except Exception as exc:                          # noqa: BLE001
        rep.check("evaluate_page_level", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    try:
        out = extract_test_features(model, metadata, label_map, device,
                                    batch_size=16, num_workers=0,
                                    page_workers=pw, segment_workers=pw,
                                    apply_drop=False, seq_len=seq_len)
        n = out['probs'].shape[0]
        rep.check("extract_test_features 概率矩阵", out['probs'].shape == (n, len(writers)),
                  f"{out['probs'].shape}")
        rep.check("静态特征 512 维", out['static_feats'] is not None
                  and out['static_feats'].shape == (n, 512),
                  f"{None if out['static_feats'] is None else out['static_feats'].shape}")
        rep.check("动态特征 256 维", out['dynamic_feats'] is not None
                  and out['dynamic_feats'].shape == (n, 256),
                  f"{None if out['dynamic_feats'] is None else out['dynamic_feats'].shape}")
        rep.check("融合特征 768 维", out['fused_feats'] is not None
                  and out['fused_feats'].shape == (n, 768),
                  f"{None if out['fused_feats'] is None else out['fused_feats'].shape}")
        rep.check("标签与页面数一致",
                  len(out['labels']) == n and len(out['pages']) == n)
    except Exception as exc:                          # noqa: BLE001
        rep.check("extract_test_features", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def check_trainer_e2e(rep, metadata, seq_len, work_dir='_smoke_out',
                      modes=('static', 'dynamic', 'dual')):
    print("\n[10] ★ 训练器端到端（小样本，每条路径各 1 个 epoch）")
    from src.training.trainer_dual import train_dual

    base = {
        'mode': 'dual',
        'batch_size': 8,
        'num_workers': 0,
        'page_workers': 0,
        'segment_workers': 0,
        'seq_len': seq_len,
        'dynamic_speed_clip': 10.0,
        'feature_dim': 512,
        'in_chans': 64,
        'drop_path': 0.0,
        'dropout_head': 0.1,
        'dropout_classifier': 0.0,
        'fusion_dropout': 0.1,
        'dynamic_dropout': 0.0,
        'pretrained': False,
        'pretrained_path': None,
        'static_init_path': None,
        'pooling': 'attention',
        'pos_encoding': 'sinusoidal',
        'warmup_epochs': 0,
        'eval_frequency': 1,
        'grad_clip': 5.0,
        'weight_decay': 0.05,
        'apply_drop': True,
        'resume': False,
        'seed': 0,
        'metadata_file': metadata,
        'checkpoint': {'save_dir': os.path.join(work_dir, 'checkpoints'),
                       'prefix': 'smoke'},
        'logging': {'log_dir': os.path.join(work_dir, 'logs'),
                    'log_file': 'smoke_log.csv'},
    }

    for mode in modes:
        cfg = dict(base)
        cfg['mode'] = mode
        cfg['checkpoint'] = dict(base['checkpoint'], prefix=f'smoke_{mode}')
        cfg['logging'] = dict(base['logging'], log_file=f'smoke_{mode}.csv')
        # 短跑到能验证回路即可
        cfg['stages'] = [
            {'name': 'e2e_stage1', 'epochs': 1, 'lr': 1e-3,
             'freeze_static': mode == 'dual', 'static_lr_scale': 0.0},
        ]
        try:
            res = train_dual(cfg)
            rep.check(f"mode={mode} 训练回路跑通", res['best_page_acc'] >= 0.0,
                      f"best_page_acc={res['best_page_acc']:.4f}")
            rep.check(f"mode={mode} 检查点已落盘",
                      os.path.exists(res['best_path']), res['best_path'])
        except Exception as exc:                      # noqa: BLE001
            rep.check(f"mode={mode} 训练回路", False,
                      f"{type(exc).__name__}: {exc}")
            traceback.print_exc()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', type=str,
                    default='data/features/metadata_small.csv')
    ap.add_argument('--ckpt', type=str,
                    default='outputs/checkpoints/convnext_best_9617.pth')
    ap.add_argument('--seq_len', type=int, default=200)
    ap.add_argument('--smoke_writers', type=int, default=3,
                    help='端到端检查用的作者数（子集 metadata）')
    ap.add_argument('--train_pages', type=int, default=2,
                    help='子集中每个作者用于训练的页面数')
    ap.add_argument('--test_pages', type=int, default=1,
                    help='子集中每个作者用于测试的页面数')
    ap.add_argument('--skip_e2e', action='store_true',
                    help='跳过第 10 项端到端训练（最快）')
    ap.add_argument('--skip_slow', action='store_true',
                    help='跳过第 8-10 项所有需要真实数据加载的检查')
    ap.add_argument('--no_mp', action='store_true',
                    help='强制单进程（Windows 建议开启）')
    ap.add_argument('--skip_static_variants', action='store_true',
                    help='跳过第 11 项 A0/A1 静态流水线开关检查')
    ap.add_argument('--skip_multi', action='store_true',
                    help='跳过第 13 项共享管道多模型训练检查')
    ap.add_argument('--work_dir', type=str, default='_smoke_out',
                    help='冒烟测试的临时输出目录（会在开始时清空）')
    ap.add_argument('--threads', type=int, default=0,
                    help='限制 torch CPU 线程数（0 = 不限制）。机器同时在跑别的任务时建议设 2-4')
    ap.add_argument('--e2e_only', action='store_true',
                    help='只跑第 10 项端到端训练（用最小子集），其余全部跳过')
    ap.add_argument('--modes', type=str, default='static,dynamic,dual',
                    help='第 10 项要验证的模式，逗号分隔')
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"🔧 torch CPU 线程数限制为 {args.threads}")

    if args.e2e_only:
        # 极小子集：确认三种模式的训练回路都能跑通即可
        args.skip_slow = False
        args.smoke_writers = 1
        args.train_pages = 1
        args.test_pages = 1

    no_mp = args.no_mp or sys.platform == 'win32'

    # 冒烟测试输出写到一个全新目录，避免污染 outputs/ 与真实训练产物
    import shutil
    if os.path.isdir(args.work_dir):
        shutil.rmtree(args.work_dir, ignore_errors=True)
    os.makedirs(args.work_dir, exist_ok=True)

    # 端到端检查用极小子集，避免 CPU 上跑成小时级
    smoke_meta = os.path.join(args.work_dir, 'metadata_smoke.csv')
    if not args.skip_slow:
        smoke_meta = make_subset_metadata(
            args.metadata, smoke_meta, n_writers=args.smoke_writers,
            train_pages_per_writer=args.train_pages,
            test_pages_per_writer=args.test_pages,
        )

    print("=" * 74)
    print("双路融合 —— 本地冒烟测试")
    print(f"元数据（子集，用于端到端检查） = {smoke_meta}")
    print(f"临时输出目录 = {os.path.abspath(args.work_dir)}")
    print(f"单进程模式 = {no_mp}")
    print("=" * 74)

    rep = Report()
    if not check_imports(rep):
        rep.summary()
        sys.exit(1)

    if args.e2e_only:
        print("\n[2-7] 已按 --e2e_only 跳过（仅验证端到端训练回路）")
    else:
        check_dynamic_branch(rep)
        check_static_equivalence(rep)
        check_three_modes(rep)
        check_freeze(rep)
        check_load_static_weights(rep, args.ckpt)
        check_optimizer_stages(rep)
        if not args.skip_static_variants:
            check_static_variants(rep)
        check_arch_switch(rep)
        check_dcnn_fidelity(rep)

    if args.skip_slow:
        print("\n[8-10] 已按 --skip_slow 跳过真实数据检查")
    else:
        if not args.e2e_only:
            check_dataset(rep, smoke_meta, args.seq_len, no_mp)
            check_evaluator(rep, smoke_meta, args.seq_len, no_mp)
        if args.skip_e2e:
            print("\n[10] 已按 --skip_e2e 跳过端到端训练")
        else:
            e2e_modes = tuple(m.strip() for m in args.modes.split(',') if m.strip())
            check_trainer_e2e(rep, smoke_meta, args.seq_len, args.work_dir,
                              modes=e2e_modes)
        if not args.skip_multi:
            check_trainer_multi(rep, smoke_meta, args.seq_len, args.work_dir)

    ok = rep.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
