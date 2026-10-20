"""
伪时序（pseudo-temporal）动态特征提取。

核心思想
--------
CASIA-OLHWDB 公开版没有时间戳，但 Anoto 笔以近似恒定频率采样坐标点。
在等间隔采样假设下，相邻点的欧氏距离正比于瞬时书写速度，三点连线的
方向变化量正比于瞬时曲率。因此**点序列的顺序本身**就携带了书写动态信息。

本模块输出固定长度 (seq_len, 2) 的动态特征序列：
    通道 0: 归一化伪速度 ṽ_i
    通道 1: 曲率 κ_i

与朴素实现的三个关键差异（均为致命缺陷修复）
------------------------------------------------
1. **分段提取，杜绝跨段速度尖峰**
   本仓库的伪字符轨迹里混有 -1 分段标记（`load_wptt_page` 在每个真实笔画
   之后插入，`apply_drop_segment_from_segments` 在每个保留下来的片段之间
   插入）。若直接对整条轨迹求 `np.diff`，(-1, 0) 与前一个真实点之间会产生
   一个量级 ~100-1000 的虚假"速度"，它会成为整条序列的最大值并彻底淹没
   真实的速度变化。本模块**先在 -1 处切分，再在每段内部计算特征**。

2. **角度环绕修正**
   方向角 θ = arctan2(dy, dx) 的取值域是 (-π, π]。当真实转向很小但恰好
   跨过 ±π 分支切割时（例如 179° → -179°，真实转向只有 2°），
   `np.abs(np.diff(angles))` 会得到 358° ≈ 6.25 rad 的虚假曲率。
   本模块用 `(dθ + π) % 2π - π` 把差分折回 (-π, π]。

3. **分段比例重采样，而不是整条插值**
   若把整条序列直接插值到固定长度，插值核会跨越分段边界，在边界附近
   制造出并不存在的平滑过渡。本模块按各段长度比例分配时间槽
   （最大余数法保证槽数总和恰为 seq_len），段内独立插值后再拼接。

约定
----
- -1 是分段（抬笔）标记，判定条件与 `render_trajectory_to_bitmap` 保持一致：
  仅检查 x 分量是否为 -1.0。
- 有效点少于 3 个时返回全零序列，与静态分支的退化处理保持一致。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PEN_UP",
    "split_segments",
    "segment_features",
    "allocate_slots",
    "extract_dynamic_sequence",
]

#: 抬笔 / 分段标记的 x 分量取值。
PEN_UP = -1.0

#: 曲率通道在段内首点无定义（需要两个差分步），按约定补 0。
_DEFAULT_SPEED_CLIP = 10.0


# ----------------------------------------------------------------------------
# 1. 分段
# ----------------------------------------------------------------------------

def split_segments(points, min_points: int = 2):
    """
    按 -1 标记把轨迹切分成连续片段。

    Args:
        points: (N, 2) 或可 reshape 成 (N, 2) 的数组，可能含 -1 分段标记。
        min_points: 片段最少保留多少个有效点。默认 2（恰好能算 1 个速度值）。

    Returns:
        List[np.ndarray]，每个元素形如 (K, 2)，K >= min_points，不含 -1。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return []
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points 形状应为 (N, 2)，实际为 {pts.shape}")

    is_sep = pts[:, 0] == PEN_UP

    segments = []
    start = None
    for i in range(len(pts)):
        if is_sep[i]:
            if start is not None:
                seg = pts[start:i]
                if len(seg) >= min_points:
                    segments.append(seg)
                start = None
        elif start is None:
            start = i
    if start is not None:
        seg = pts[start:]
        if len(seg) >= min_points:
            segments.append(seg)

    return segments


# ----------------------------------------------------------------------------
# 2. 段内特征
# ----------------------------------------------------------------------------

def segment_features(seg):
    """
    计算单个连续片段的伪速度与曲率，两者长度均为 K-1（K = len(seg)）。

    对齐约定（与方案文档一致）：曲率通道在首位补 0，即
        curv[0] = 0
        curv[i] = |wrap(θ_i - θ_{i-1})|   , i >= 1
    其中 θ_i 是第 i 个差分向量的方向角。

    Args:
        seg: (K, 2) 连续片段，不含 -1，K >= 2。

    Returns:
        (speed, curvature): 两个 (K-1,) 的 float64 数组。
    """
    seg = np.asarray(seg, dtype=np.float64)
    diff = np.diff(seg, axis=0)                       # (K-1, 2)
    speed = np.hypot(diff[:, 0], diff[:, 1])          # (K-1,)
    n = len(speed)

    curvature = np.zeros(n, dtype=np.float64)
    if n >= 2:
        angles = np.arctan2(diff[:, 1], diff[:, 0])   # (K-1,) in (-π, π]
        d_theta = np.diff(angles)
        # 关键修复 2：折回 (-π, π]，消除 ±π 分支切割处的虚假大曲率
        d_theta = (d_theta + np.pi) % (2.0 * np.pi) - np.pi
        curvature[1:] = np.abs(d_theta)

    return speed, curvature


# ----------------------------------------------------------------------------
# 3. 时间槽分配
# ----------------------------------------------------------------------------

def allocate_slots(weights, total: int):
    """
    按权重比例把 total 个时间槽分配给各片段，用最大余数法保证总和恰为 total。

    每个片段至少分到 1 个槽（当 total >= 片段数时），避免短片段被完全丢弃。

    Args:
        weights: 长度 n 的权重序列（此处用各段的特征长度）。
        total: 总槽数（= seq_len）。

    Returns:
        List[int]，长度 n，元素之和恰为 min(total, ...) == total（n>0 时）。
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = len(w)
    if n == 0 or total <= 0:
        return [0] * n

    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(n, dtype=np.float64)

    # 槽数比片段数还少：把槽给最长的 total 个片段，其余置 0（会被跳过）
    if total < n:
        order = np.argsort(-w, kind="stable")
        alloc = np.zeros(n, dtype=np.int64)
        alloc[order[:total]] = 1
        return alloc.tolist()

    # 先给每段留 1 个底盘，剩余 total-n 个按权重比例分配
    quota = w / w.sum() * (total - n)
    base = np.floor(quota).astype(np.int64)
    remainder = quota - base
    alloc = base + 1

    # 最大余数法补齐；由于 floor 的性质，short 必落在 [0, n)
    short = total - int(alloc.sum())
    if short > 0:
        order = np.argsort(-remainder, kind="stable")
        alloc[order[:short]] += 1

    return alloc.tolist()


# ----------------------------------------------------------------------------
# 4. 主入口
# ----------------------------------------------------------------------------

def extract_dynamic_sequence(char_points, seq_len: int = 200,
                             speed_clip: float = _DEFAULT_SPEED_CLIP,
                             min_points: int = 3,
                             dtype=np.float32):
    """
    从单个伪字符的原始点序列提取定长动态特征序列。

    Args:
        char_points: (N, 2) 点序列，可含 -1 分段标记（推荐传入带标记的
            `drop_points`，这样分支能识别拖拽增强造成的真实断裂）。
        seq_len: 输出序列长度，默认 200（与方案文档一致）。
        speed_clip: 归一化伪速度的上限截断，默认 10.0。
            归一化后均值恰为 1，取 10 相当于「比平均快 10 倍」封顶，
            防止个别极快笔画主导整个序列的尺度。设为 None 关闭截断。
        min_points: 整条轨迹有效点少于该数量时返回全零序列，默认 3。
        dtype: 输出数据类型。

    Returns:
        (seq_len, 2) 数组：通道 0 为归一化伪速度（均值≈1），通道 1 为曲率。

    Notes:
        归一化伪速度 ṽ_i = v_i / mean(v)。
        曲率本身已在 [0, π] 内，不做额外归一化（与方案文档一致）。
    """
    zero = np.zeros((seq_len, 2), dtype=dtype)

    pts = np.asarray(char_points)
    if pts.size == 0:
        return zero
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"char_points 形状应为 (N, 2)，实际为 {pts.shape}")

    # 整条轨迹的有效点数（去掉 -1 标记）
    n_valid = int(np.count_nonzero(pts[:, 0] != PEN_UP))
    if n_valid < min_points:
        return zero

    segments = split_segments(pts, min_points=2)
    if not segments:
        return zero

    # 逐段计算特征（关键修复 1：绝不在段间求差分）
    feats = [segment_features(seg) for seg in segments]
    lengths = [len(s) for s, _ in feats]

    slots = allocate_slots(lengths, seq_len)

    speed_parts = []
    curv_parts = []
    for (speed, curvature), n_slot in zip(feats, slots):
        if n_slot <= 0:
            continue
        src = np.linspace(0.0, 1.0, len(speed))
        dst = np.linspace(0.0, 1.0, n_slot)
        speed_parts.append(np.interp(dst, src, speed))
        curv_parts.append(np.interp(dst, src, curvature))

    if not speed_parts:
        return zero

    speed_seq = np.concatenate(speed_parts)
    curv_seq = np.concatenate(curv_parts)

    # 长度对齐保护：allocate_slots 保证和恰为 seq_len，此处仅作断言性质兜底
    if len(speed_seq) != seq_len:
        speed_seq = _fit_length(speed_seq, seq_len)
        curv_seq = _fit_length(curv_seq, seq_len)

    # 归一化伪速度（除以该字符的平均点间距，消除字符大小差异）
    mean_speed = float(speed_seq.mean())
    if mean_speed > 1e-6:
        speed_seq = speed_seq / mean_speed
    else:
        speed_seq = np.zeros(seq_len, dtype=np.float64)

    if speed_clip is not None:
        # 上界截断；下界恒为 0，无需处理
        speed_seq = np.clip(speed_seq, 0.0, float(speed_clip))

    out = np.stack([speed_seq, curv_seq], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype)


def _fit_length(seq, target: int):
    """把一维序列线性拉伸/截断到 target 长度（兜底路径）。"""
    seq = np.asarray(seq, dtype=np.float64).ravel()
    if len(seq) == target:
        return seq
    if len(seq) == 0:
        return np.zeros(target, dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(seq))
    dst = np.linspace(0.0, 1.0, target)
    return np.interp(dst, src, seq)
