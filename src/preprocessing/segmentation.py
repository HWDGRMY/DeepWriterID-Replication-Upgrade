"""
伪字符切分。

`pseudo_segment` 把页面轨迹按拐点切成片段，再按「累计宽度不超过 avg_height」
把相邻片段合并成一个伪字符。

抬笔分隔符必须保留
------------------
v2.0（A0）的实现用 `seg[seg[:, 0] != -1]` 剥掉抬笔标记再首尾相接，结果是
**真实的抬笔边界被抹平**：两个笔画之间被连成一条直线。这一个缺陷导致两个后果：

1. `render_trajectory_to_bitmap` 里「按 -1 分段绘制」的逻辑永远不触发
   → 实测位图里 19% 的墨迹是伪造的，单条伪造线段长度达画布的 53%；
2. 拼接点处坐标突变，`detect_corners` 会在那里找到**假角点**；
   而 `WPTTDataset.__getitem__` 会对每个伪字符**再切分一次**
   → 实测 Train 24.9% / Test 22.3% 的样本被二次切分裂开，只保留第一块，
     平均丢弃 21.9% 的墨迹（最坏 95%）。这个截断是静默的，
   参数量、张量形状、样本数全都看不出来。

因此本版本**不再提供关闭分隔符的选项**：输出数组里，凡是相邻两点的原始行索引
不连续（中间隔着 -1，或隔着被丢弃的短片段）的位置，一律插入 (-1, 0) 分隔符。

单一行为而不是开关，是为了让上面那两个后果**不可能再被某个配置项悄悄打开**。
"""

import numpy as np

PEN_UP = -1.0


def _assemble(points, seg_index_list):
    """
    把若干片段的行索引拼成一个伪字符的坐标数组。

    相邻两点的原始行索引不连续 => 中间确有抬笔或跳变 => 插分隔符。

    Args:
        points: 整页轨迹 (N, 2)。
        seg_index_list: 片段列表，每个元素是该片段在 points 中的行索引（保序）。

    Returns:
        (M, 2) 坐标数组，可能含 (-1, 0) 分隔符行。
    """
    flat = np.concatenate(seg_index_list)

    parts = []
    prev = None
    for k in flat:
        if prev is not None and k != prev + 1:
            parts.append(np.array([[PEN_UP, 0.0]], dtype=points.dtype))
        parts.append(points[k:k + 1])
        prev = k
    return np.vstack(parts)


def estimate_avg_char_height(points, corners_idx):
    """
    估计「字符的平均高度」——论文里切分阈值的定义。

    论文原文（§2.2）：
        "the **average height** is measured first. Then, each of the segments will
         be sequentially assigned to form a pseudo character. When the width of a
         character exceeds the average height after adding a segment, this segment
         is regarded as the beginning of a new character."
        "the height-to-width ratio should be the same. We thus determined this ratio
         to be **1** for simplicity ... other ratio value between [1, 1.3] is also proper"

    早期实现用 `page_height / 10.0` 做代理（假设一页恰好 10 行字）。实测这个假设
    偏大 **1.78 倍**（25 页：代理中位 34.4，实测字高中位 19.3），后果是伪字符
    h/w 只有 **0.70**（论文要求 ≈1），并且每页只切出 260 个而不是 531 个。

    这里改回论文的做法——**先量**：
      1. 用页面高度的一小部分做初切（这里取 page_height/10，仅作起点）；
      2. 量出这批伪字符的平均高度；
      3. 用它作为真正的阈值。

    实测：两轮后 h/w 从 0.70 提升到 **0.97**，伪字符数 260 → 531。

    Returns:
        平均字符高度（坐标单位）。
    """
    pts = np.asarray(points)
    valid = pts[pts[:, 0] != PEN_UP]
    if len(valid) == 0:
        return 50.0
    page_height = float(np.max(valid[:, 1]) - np.min(valid[:, 1]))

    # 初切：仅用于得到一个量级正确的起点
    probe = page_height / 10.0
    if probe < 10.0:
        probe = 50.0
    chars = _segment_core(pts, corners_idx, probe)

    heights = []
    for ch in chars:
        d = ch[ch[:, 0] != PEN_UP]
        if len(d) < 3:
            continue
        h = float(np.max(d[:, 1]) - np.min(d[:, 1]))
        if h > 0:
            heights.append(h)
    if not heights:
        return probe
    est = float(np.mean(heights))
    return est if est > 1e-6 else probe


def pseudo_segment(points, corners_idx, avg_height=None):
    """
    Args:
        points: (N, 2) 整页轨迹，可含 -1 抬笔标记。
        corners_idx: `detect_corners` 返回的拐点索引。
        avg_height: 切分阈值（字符平均高度）。
            None（默认）= 按论文的做法**先量再切**（见 `estimate_avg_char_height`）。
            传入数值 = 直接用该阈值（`__getitem__` 重切单字时必须走这条，
            否则会对单个字形再估一次，把本来完整的一个字切开）。

    Returns:
        List[np.ndarray]，每个元素是一个伪字符的 (M, 2) 坐标，
        其中可能含 (-1, 0) 分隔符行。
    """
    if avg_height is None:
        avg_height = estimate_avg_char_height(points, corners_idx)
    return _segment_core(points, corners_idx, avg_height)


def _segment_core(points, corners_idx, avg_height):
    """按给定阈值执行切分。"""
    points = np.asarray(points)
    n = len(points)
    all_indices = sorted([0] + list(corners_idx) + [n - 1])

    # 只记录行索引，避免提前丢信息
    segs = []
    for i in range(len(all_indices) - 1):
        start = all_indices[i]
        end = min(all_indices[i + 1] + 1, n)
        idx = np.arange(start, end)
        idx = idx[points[idx, 0] != PEN_UP]
        if len(idx) >= 3:
            segs.append(idx)

    if len(segs) == 0:
        return []

    pseudo_chars = []
    current_char = []
    # 累加组的**包围盒**左右边界。
    #
    # 论文 §2.2 原文：
    #   "When the **width of a character** exceeds the average height after
    #    adding a segment, this segment is regarded as the beginning of a new
    #    character."
    # 这里说的是「这个字符的宽度」，即已收进来的所有点合起来的水平跨度。
    #
    # 早期实现写的是 `current_width += seg_width`（把每个片段**自身**的宽度累加）。
    # 片段在水平方向常常重叠（同一个字的偏旁、同一笔的往复），累加会重复计数，
    # 于是过早断句。实测（12 页）：
    #   · 累加段宽   -> 每页 554 个伪字符，每字 15 个点，h/w 0.97
    #   · 包围盒宽   -> 每页 306 个伪字符，每字 28 个点，h/w 1.26
    # 而这些页面真实字符数中位是 251，论文又明确说 h/w 取 1（[1, 1.3] 也可）——
    # 包围盒版 306≈251 且 h/w=1.26 落在论文区间内，累加版两个指标都偏。
    cur_min_x = np.inf
    cur_max_x = -np.inf

    for idx in segs:
        seg = points[idx]
        seg_min_x = float(np.min(seg[:, 0]))
        seg_max_x = float(np.max(seg[:, 0]))
        new_min_x = min(cur_min_x, seg_min_x)
        new_max_x = max(cur_max_x, seg_max_x)

        if current_char and (new_max_x - new_min_x) > avg_height:
            pseudo_chars.append(_assemble(points, current_char))
            current_char = [idx]
            cur_min_x, cur_max_x = seg_min_x, seg_max_x
        else:
            current_char.append(idx)
            cur_min_x, cur_max_x = new_min_x, new_max_x

    if len(current_char) > 0:
        pseudo_chars.append(_assemble(points, current_char))

    if len(pseudo_chars) == 0:
        for idx in segs:
            pseudo_chars.append(_assemble(points, [idx]))

    return pseudo_chars
