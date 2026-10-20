"""
伪时序动态特征提取的正确性验证。

本脚本**只依赖 numpy**（真实数据部分额外依赖本仓库的纯 numpy 预处理模块），
不需要 torch / GPU，可以在任何机器上几秒内跑完。云端开跑前建议先执行一次。

用法:
    python scripts/verify_dynamic.py
    python scripts/verify_dynamic.py --n_real 200 --seq_len 200

覆盖的 8 项检查:
    1. 匀速直线 → 归一化伪速度恒为 1、曲率恒为 0
    2. 分段边界不产生虚假速度尖峰（对比朴素实现）
    3. 转向角跨 ±π 分支切割时的曲率正确性（对比朴素实现）
    4. 时间槽分配：总和恰为 seq_len、每段至少 1 槽
    5. 输出形状 / 数据类型 / 有限性 / 确定性
    6. 退化输入（空、单点、全 -1）返回全零而不报错
    7. 真实 .wptt 数据端到端统计
    8. 朴素实现 vs 修复实现的量化对比
"""

from __future__ import annotations

import os
import sys
import argparse
import glob

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Windows 控制台默认 GBK，会让中文报告变成乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.features.dynamic import (  # noqa: E402
    PEN_UP,
    allocate_slots,
    extract_dynamic_sequence,
    segment_features,
    split_segments,
)


# ---------------------------------------------------------------------------
# 朴素实现：仅用于第 2/3/8 项检查的对照，故意保留方案文档里的原始写法
# ---------------------------------------------------------------------------

def naive_extract(char_points, seq_len=200):
    pts = np.asarray(char_points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if len(pts) < 3:
        return np.zeros((seq_len, 2), dtype=np.float32)

    diffs = np.diff(pts, axis=0)
    speeds = np.sqrt((diffs ** 2).sum(axis=1))
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    curvatures = np.abs(np.diff(angles))
    curvatures = np.concatenate([[0], curvatures])

    orig = np.linspace(0, 1, len(speeds))
    target = np.linspace(0, 1, seq_len)
    sp = np.interp(target, orig, speeds)
    cv = np.interp(target, orig, curvatures)
    if sp.mean() > 1e-6:
        sp = sp / sp.mean()
    return np.stack([sp, cv], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 检查框架
# ---------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.rows = []

    def check(self, name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        self.rows.append((tag, name, detail))
        print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
        return ok

    def summary(self):
        print("\n" + "=" * 74)
        print(f"结果: {self.passed} 项通过, {self.failed} 项失败")
        print("=" * 74)
        return self.failed == 0


# ---------------------------------------------------------------------------
# 1. 匀速直线
# ---------------------------------------------------------------------------

def check_uniform_line(rep, seq_len):
    print("\n[1] 匀速直线：归一化伪速度应恒为 1，曲率应恒为 0")
    pts = np.stack([np.arange(50, dtype=np.float64) * 3.0,
                    np.zeros(50, dtype=np.float64)], axis=1)
    seq = extract_dynamic_sequence(pts, seq_len=seq_len)

    speed, curv = seq[:, 0], seq[:, 1]
    rep.check("形状正确", seq.shape == (seq_len, 2), f"shape={seq.shape}")
    rep.check("归一化伪速度恒为 1", np.allclose(speed, 1.0, atol=1e-5),
              f"min={speed.min():.6f} max={speed.max():.6f}")
    rep.check("曲率恒为 0", np.allclose(curv, 0.0, atol=1e-9),
              f"max={curv.max():.3e}")


# ---------------------------------------------------------------------------
# 2. 分段边界不产生速度尖峰
# ---------------------------------------------------------------------------

def check_no_boundary_spike(rep, seq_len):
    print("\n[2] 分段边界：不应产生跨段虚假速度尖峰")
    seg_a = np.stack([np.arange(50, dtype=np.float64),
                      np.zeros(50, dtype=np.float64)], axis=1)
    seg_b = np.stack([np.arange(500, 550, dtype=np.float64),
                      np.full(50, 300.0)], axis=1)
    sep = np.array([[PEN_UP, 0.0]])
    traj = np.vstack([seg_a, sep, seg_b])

    # 段间真实点间距：段内为 1.0，跨段（若忽略 -1）约 584
    seq = extract_dynamic_sequence(traj, seq_len=seq_len, speed_clip=None)
    naive = naive_extract(traj, seq_len=seq_len)

    fixed_max = float(seq[:, 0].max())
    naive_max = float(naive[:, 0].max())

    rep.check("修复实现无尖峰（归一化速度 <= 1.0 + eps）",
              fixed_max <= 1.0 + 1e-5, f"max={fixed_max:.6f}")
    rep.check("朴素实现确实存在巨大尖峰（说明该修复必要）",
              naive_max > 5.0, f"朴素 max={naive_max:.2f}")
    rep.check("朴素尖峰比修复值大一个数量级以上",
              naive_max / max(fixed_max, 1e-9) > 10.0,
              f"倍数={naive_max / max(fixed_max, 1e-9):.1f}x")

    # 分段识别本身
    segs = split_segments(traj)
    rep.check("正确切分为 2 段", len(segs) == 2, f"len={len(segs)}")


# ---------------------------------------------------------------------------
# 3. 角度环绕
# ---------------------------------------------------------------------------

def check_angle_wrap(rep, seq_len):
    print("\n[3] 角度环绕：跨 ±π 分支切割的小转向不应被放大")
    two_deg = np.deg2rad(2.0)

    for name, a0_deg, a1_deg in [("+179° → -179°", 179.0, -179.0),
                                 ("-179° → +179°", -179.0, 179.0)]:
        a0, a1 = np.deg2rad(a0_deg), np.deg2rad(a1_deg)
        p0 = np.array([0.0, 0.0])
        p1 = p0 + np.array([np.cos(a0), np.sin(a0)])
        p2 = p1 + np.array([np.cos(a1), np.sin(a1)])
        pts = np.stack([p0, p1, p2])

        _, curv = segment_features(pts)
        got = float(curv[1])                 # curv[0] 约定补 0
        naive = abs(a1_deg - a0_deg) * np.pi / 180.0

        rep.check(f"{name}: 曲率 ≈ 2° ({two_deg:.4f} rad)",
                  abs(got - two_deg) < 1e-6, f"实际={got:.6f}")
        rep.check(f"{name}: 朴素实现会放大到 358°",
                  naive > 6.0, f"朴素={naive:.4f} rad")

    # 真实 90° 转向应被正确保留
    pts90 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    _, curv90 = segment_features(pts90)
    rep.check("90° 转向曲率 ≈ π/2", abs(float(curv90[1]) - np.pi / 2) < 1e-9,
              f"实际={float(curv90[1]):.6f}")


# ---------------------------------------------------------------------------
# 4. 时间槽分配
# ---------------------------------------------------------------------------

def check_allocate_slots(rep):
    print("\n[4] 时间槽分配：总和恰为 seq_len，每段至少 1 槽")
    cases = [
        ([10, 20, 30], 200),
        ([1], 200),
        ([1, 1, 1, 1, 1], 200),
        ([5, 500, 7], 200),
        ([1, 2], 3),
        ([1, 1, 1, 1, 1, 1], 3),      # 槽数少于段数
        ([0, 0, 0], 200),             # 全零权重退化为均匀
    ]
    ok_sum, ok_min = True, True
    for weights, total in cases:
        alloc = allocate_slots(weights, total)
        s = sum(alloc)
        if s != total:
            ok_sum = False
            rep.check(f"weights={weights} total={total} 总和", False,
                      f"sum={s}")
        # 仅当槽数够时要求每段至少 1
        if total >= len(weights) and min(alloc) < 1:
            ok_min = False
            rep.check(f"weights={weights} total={total} 最小槽", False,
                      f"min={min(alloc)}")
    rep.check("所有用例槽数总和恰为 total", ok_sum)
    rep.check("槽数充足时每段至少 1 槽", ok_min)


# ---------------------------------------------------------------------------
# 5. 形状 / 类型 / 有限性 / 确定性
# ---------------------------------------------------------------------------

def check_shape_dtype_determinism(rep, seq_len):
    print("\n[5] 输出形状 / 数据类型 / 有限性 / 确定性")
    rng = np.random.default_rng(0)
    for n in [3, 4, 10, 200, 500]:
        pts = np.cumsum(rng.normal(0, 2.0, size=(n, 2)), axis=0)
        seq = extract_dynamic_sequence(pts, seq_len=seq_len)
        assert seq.shape == (seq_len, 2), f"n={n} shape={seq.shape}"
        assert seq.dtype == np.float32, f"n={n} dtype={seq.dtype}"
        assert np.isfinite(seq).all(), f"n={n} 存在非有限值"
    rep.check("多种输入长度均输出 (seq_len, 2) float32 且有限", True)

    pts = np.cumsum(rng.normal(0, 2.0, size=(120, 2)), axis=0)
    a = extract_dynamic_sequence(pts, seq_len=seq_len)
    b = extract_dynamic_sequence(pts, seq_len=seq_len)
    rep.check("确定性：两次调用逐元素相同", np.array_equal(a, b))

    # 速度归一化后均值应接近 1（未被 clip 拉低时应恰为 1）
    seq = extract_dynamic_sequence(pts, seq_len=seq_len, speed_clip=None)
    rep.check("归一化后伪速度均值 ≈ 1",
              abs(float(seq[:, 0].mean()) - 1.0) < 1e-4,
              f"mean={float(seq[:, 0].mean()):.8f}")

    # 曲率上界：折回后的 |Δθ| <= π
    rep.check("曲率通道取值落在 [0, π]",
              float(seq[:, 1].min()) >= 0.0 and float(seq[:, 1].max()) <= np.pi + 1e-6,
              f"[{seq[:, 1].min():.4f}, {seq[:, 1].max():.4f}]")


# ---------------------------------------------------------------------------
# 6. 退化输入
# ---------------------------------------------------------------------------

def check_degenerate(rep, seq_len):
    print("\n[6] 退化输入应安全返回全零")
    cases = {
        "空数组": np.zeros((0, 2)),
        "单个点": np.array([[3.0, 4.0]]),
        "两个点": np.array([[3.0, 4.0], [5.0, 6.0]]),
        "全是 -1 标记": np.array([[PEN_UP, 0.0], [PEN_UP, 0.0], [PEN_UP, 0.0]]),
        "两段各只有一个点": np.array([[0.0, 0.0], [PEN_UP, 0.0], [9.0, 9.0]]),
        "无有效点（-1 与单点混合）": np.array([[PEN_UP, 0.0], [1.0, 1.0],
                                                [PEN_UP, 0.0], [2.0, 2.0]]),
    }
    ok = True
    for name, pts in cases.items():
        try:
            seq = extract_dynamic_sequence(pts, seq_len=seq_len)
            if seq.shape != (seq_len, 2) or seq.any():
                ok = False
                rep.check(name, False, f"shape={seq.shape} any={seq.any()}")
        except Exception as exc:                      # noqa: BLE001
            ok = False
            rep.check(name, False, f"抛出 {type(exc).__name__}: {exc}")
    rep.check("所有退化输入均返回全零 (seq_len, 2) 而不报错", ok)

    # 恰好构成一个 2 点连续段（1 个速度值、0 个曲率）应给出有限的合法输出，
    # 而不是被误判为退化输入
    pts2 = np.array([[0.0, 0.0], [3.0, 4.0], [PEN_UP, 0.0], [9.0, 9.0]])
    seq2 = extract_dynamic_sequence(pts2, seq_len=seq_len)
    rep.check("单段 2 点轨迹给出有限非负输出",
              bool(np.isfinite(seq2).all()) and float(seq2.min()) >= 0.0
              and float(seq2[:, 0].max()) > 0.0,
              f"max={float(seq2[:, 0].max()):.4f}")

    # 一维输入应能被 reshape
    try:
        seq = extract_dynamic_sequence(np.array([0.0, 0.0, 1.0, 0.0, 2.0, 0.0]),
                                       seq_len=seq_len)
        ok1d = seq.shape == (seq_len, 2) and np.isfinite(seq).all()
    except Exception as exc:                          # noqa: BLE001
        ok1d = False
        rep.check("一维输入", False, str(exc))
    rep.check("一维扁平输入可正常处理", ok1d)


# ---------------------------------------------------------------------------
# 7. 真实数据端到端
# ---------------------------------------------------------------------------

def check_real_data(rep, n_real, seq_len):
    print(f"\n[7] 真实 .wptt 数据端到端（最多 {n_real} 个伪字符）")
    try:
        from src.data.loader import load_wptt_page
        from src.preprocessing.corner import detect_corners
        from src.preprocessing.segmentation import pseudo_segment
        from src.augmentation.drop_segment import apply_drop_segment
    except Exception as exc:                          # noqa: BLE001
        rep.check("导入预处理模块", False, str(exc))
        return

    root = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    files = sorted(glob.glob(os.path.join(root, "WPTT2.0-Train", "*.wptt")))
    if not files:
        print("  [SKIP] 未找到本地 .wptt 文件，跳过真实数据检查")
        return

    seqs = []
    n_drop_spike = 0
    for fp in files[:3]:
        points = load_wptt_page(fp)
        corners = detect_corners(points, k=2, threshold=3)
        chars = pseudo_segment(points, corners)
        for ch in chars[: max(1, n_real // 3)]:
            drop_points = apply_drop_segment(ch)
            seq = extract_dynamic_sequence(drop_points, seq_len=seq_len)
            seqs.append(seq)
            # 检查是否仍有异常尖峰
            if float(seq[:, 0].max()) > 10.0 + 1e-5:
                n_drop_spike += 1
            if len(seqs) >= n_real:
                break
        if len(seqs) >= n_real:
            break

    if not seqs:
        rep.check("生成动态序列", False, "未生成任何样本")
        return

    arr = np.stack(seqs)
    print(f"       样本数={len(seqs)}  形状={arr.shape}")
    print(f"       速度: 均值={arr[:, :, 0].mean():.4f}  "
          f"标准差={arr[:, :, 0].std():.4f}  最大={arr[:, :, 0].max():.3f}")
    print(f"       曲率: 均值={arr[:, :, 1].mean():.4f}  "
          f"标准差={arr[:, :, 1].std():.4f}  最大={arr[:, :, 1].max():.3f}")

    rep.check("真实数据输出形状正确", arr.shape == (len(seqs), seq_len, 2))
    rep.check("真实数据全部有限", bool(np.isfinite(arr).all()))
    rep.check(f"归一化速度不超过截断上限", float(arr[:, :, 0].max()) <= 10.0 + 1e-5,
              f"max={float(arr[:, :, 0].max()):.4f}")
    rep.check("曲率落在 [0, π]", float(arr[:, :, 1].max()) <= np.pi + 1e-6,
              f"max={float(arr[:, :, 1].max()):.4f}")
    rep.check("特征非常数（具备区分度）",
              float(arr[:, :, 0].std()) > 1e-3 and float(arr[:, :, 1].std()) > 1e-4,
              f"std=({arr[:, :, 0].std():.4f}, {arr[:, :, 1].std():.4f})")


# ---------------------------------------------------------------------------
# 8. 朴素 vs 修复 的量化对比
# ---------------------------------------------------------------------------

def check_naive_vs_fixed(rep, seq_len, n=60):
    print("\n[8] 朴素实现 vs 修复实现（真实数据，量化缺陷影响）")
    try:
        from src.data.loader import load_wptt_page
        from src.preprocessing.corner import detect_corners
        from src.preprocessing.segmentation import pseudo_segment
        from src.augmentation.drop_segment import apply_drop_segment
    except Exception as exc:                          # noqa: BLE001
        rep.check("导入预处理模块", False, str(exc))
        return

    root = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    files = sorted(glob.glob(os.path.join(root, "WPTT2.0-Train", "*.wptt")))
    if not files:
        print("  [SKIP] 未找到本地 .wptt 文件，跳过对比检查")
        return

    naive_maxes, fixed_maxes = [], []
    naive_curv_maxes, fixed_curv_maxes = [], []
    for fp in files[:2]:
        points = load_wptt_page(fp)
        chars = pseudo_segment(points, detect_corners(points, k=2, threshold=3))
        for ch in chars:
            if len(naive_maxes) >= n:
                break
            dp = apply_drop_segment(ch)
            nv = naive_extract(dp, seq_len=seq_len)
            fx = extract_dynamic_sequence(dp, seq_len=seq_len, speed_clip=None)
            naive_maxes.append(float(nv[:, 0].max()))
            fixed_maxes.append(float(fx[:, 0].max()))
            naive_curv_maxes.append(float(nv[:, 1].max()))
            fixed_curv_maxes.append(float(fx[:, 1].max()))
        if len(naive_maxes) >= n:
            break

    if not naive_maxes:
        print("  [SKIP] 未收集到样本")
        return

    nm = float(np.mean(naive_maxes))
    fm = float(np.mean(fixed_maxes))
    ncm = float(np.mean(naive_curv_maxes))
    fcm = float(np.mean(fixed_curv_maxes))
    print(f"       归一化速度最大值（均值）: 朴素={nm:.2f}  修复={fm:.2f}  放大={nm/max(fm,1e-9):.1f}x")
    print(f"       曲率最大值（均值）      : 朴素={ncm:.4f}  修复={fcm:.4f}  放大={ncm/max(fcm,1e-9):.1f}x")

    rep.check("修复后速度最大值显著低于朴素实现", fm < nm,
              f"{fm:.2f} < {nm:.2f}")
    rep.check("修复后的曲率不超出 π 的理论上界", fcm <= np.pi + 1e-6,
              f"{fcm:.4f} <= π")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--n_real", type=int, default=60)
    args = ap.parse_args()

    print("=" * 74)
    print("伪时序动态特征提取 — 正确性验证")
    print(f"seq_len = {args.seq_len}")
    print("=" * 74)

    rep = Report()
    check_uniform_line(rep, args.seq_len)
    check_no_boundary_spike(rep, args.seq_len)
    check_angle_wrap(rep, args.seq_len)
    check_allocate_slots(rep)
    check_shape_dtype_determinism(rep, args.seq_len)
    check_degenerate(rep, args.seq_len)
    check_real_data(rep, args.n_real, args.seq_len)
    check_naive_vs_fixed(rep, args.seq_len)

    ok = rep.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
