import numpy as np
import random
from src.preprocessing.corner import detect_corners
from src.preprocessing.segmentation import pseudo_segment


def get_segments(points, threshold=3):
    """
    检测拐点并切分片段，返回片段索引列表 [(start, end), ...]。
    只做一次，可缓存。片段判定与原 apply_drop_segment 一致：
    过滤掉 (-1, 0) 后，有效点数 >= 3 才保留。
    """
    corners = detect_corners(points, k=2, threshold=threshold)
    all_indices = sorted([0] + list(corners) + [len(points) - 1])
    segments = []
    for i in range(len(all_indices) - 1):
        start, end = all_indices[i], all_indices[i + 1]
        seg = points[start:end + 1]
        if seg.ndim == 1:
            seg = seg.reshape(-1, 2)
        seg = seg[seg[:, 0] != -1]
        if len(seg) >= 3:
            segments.append((start, end))
    return segments


def apply_drop_segment_from_segments(points, segments, max_remove=1):
    """
    按论文 §3 "DropSegment Implementation" 的规则随机删片段。

    论文原文
    --------
        "If a character contains **m' >= m** (m' >> m) segments, a **random number
         r ∈ [0, m]** of the segments will be removed; the removed segments are
         also randomly chosen. Our DropSegment using this setting not only generates
         a considerable quantity of samples but also **prevents too much information
         loss**."

    即：
      · 只处理片段数 **m' >= m** 的字符（片段太少的字符跳过）；
      · 每次删 **r ~ Uniform{0..m}** 段，**m 是固定的小常数**，不是百分比；
      · "prevents too much information loss" 说明 m 故意取得很小。

    历史上的两种错误实现
    --------------------
    1. 复刻仓库的 `max_remove_ratio=0.5`：按**百分比**删。论文里没有这个比例，
       只有固定上限 m。对片段多的字符会删掉一半，远多于论文。
    2. 复刻仓库的 `if len(segments) <= 3: return points`：硬编码阈值 3。
       论文的条件是 `m' >= m`，阈值应当就是 m 本身。

    参数
    ----
    max_remove : 论文里的 m。默认 1。
                 · 只在 `m' >= m` 时执行（论文条件）；
                 · 删除数 r 从 {0..m} 均匀取，且永不删光（否则夹到 m'-1）。

    与论文仍存的差异
    ----------------
    DropSegment 的**策略**已与论文一致（固定 m、r ~ U{0..m}、不删光、
    m' >= m 才执行）。剩下的差异在 **m' 本身**——每个伪字符的片段数。
    论文的例子是一个 3 笔汉字含 2+3+4 = **9 个片段**；我们的拐点检测
    （`detect_corners(k=2, threshold=3)`）给出的中位只有 **2 个**。
    所以论文的 m' ≫ m 在我们这里并不成立，m 只能取 1。
    要缩小这个差距需要换拐点检测，那是 `src/preprocessing/corner.py` 的事，
    不属于 DropSegment。
    """
    n_seg = len(segments)
    # 论文条件 m' >= m
    if n_seg < max_remove:
        return points

    # 论文：r ~ Uniform{0..m}；同时「永不删光」（"we would never remove all the
    # segments of a character"），所以上限再夹到 n_seg - 1。
    hi = min(max_remove, n_seg - 1)
    if hi <= 0:
        return points
    remove_count = random.randint(0, hi)
    if remove_count == 0:
        return points
    remove_indices = set(random.sample(range(n_seg), remove_count))

    new_points = []
    for i, (start, end) in enumerate(segments):
        if i not in remove_indices:
            seg = points[start:end + 1]
            if seg.ndim == 1:
                seg = seg.reshape(-1, 2)
            seg = seg[seg[:, 0] != -1]
            new_points.extend(seg)
            # 与原逻辑保持一致：仅在原始片段不是最后一个时插入分隔符
            if i < len(segments) - 1:
                new_points.append((-1.0, 0.0))

    if len(new_points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(new_points, dtype=np.float32)


def apply_drop_segment(points, max_remove=1):
    """兼容旧接口：内部先检测片段，再删除。"""
    segments = get_segments(points)
    return apply_drop_segment_from_segments(points, segments, max_remove)