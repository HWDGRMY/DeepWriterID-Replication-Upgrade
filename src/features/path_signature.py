"""
路径签名特征提取。

关键修复：
- render_trajectory_to_bitmap 按 -1 分段绘制，避免多笔画被一条直线连接
- 数据流中的 -1 是笔画分隔符（抬笔点），必须正确处理
"""

import numpy as np
import cv2
import torch
import signatory


def render_trajectory_to_bitmap(points, target_size=54):
    """
    把轨迹渲染成位图。

    关键点：按 -1（抬笔点）分段绘制，避免多笔画之间被虚假直线连接。
    例如「二」的两横，如果一次性画所有点，中间会多一条竖线；分段绘制后只有两条横。

    返回 (bitmap, norm_points)：
    - bitmap: 归一化到 [0, 1] 的 target_size × target_size 灰度图
    - norm_points: 所有有效点（不含 -1）归一化后的整数坐标，保序
    """
    points = np.array(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 2)

    is_pen_up = points[:, 0] == -1
    down_points = points[~is_pen_up]
    if len(down_points) < 3:
        return np.zeros((target_size, target_size), dtype=np.float32), down_points

    # 全局包围盒
    min_x, max_x = np.min(down_points[:, 0]), np.max(down_points[:, 0])
    min_y, max_y = np.min(down_points[:, 1]), np.max(down_points[:, 1])
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    scale = min((target_size - 4) / width, (target_size - 4) / height)

    def _normalize(seg):
        seg = np.array(seg, dtype=np.float32)
        seg[:, 0] = (seg[:, 0] - min_x) * scale + 2
        seg[:, 1] = (seg[:, 1] - min_y) * scale + 2
        return seg.astype(int)

    # 按 -1 分段绘制
    img = np.full((target_size, target_size), 255, dtype=np.uint8)
    cur_seg = []
    for pt in points:
        if pt[0] == -1:
            if len(cur_seg) >= 2:
                norm_seg = _normalize(cur_seg)
                cv2.polylines(img, [norm_seg], isClosed=False, color=0, thickness=1)
            cur_seg = []
        else:
            cur_seg.append(pt)
    if len(cur_seg) >= 2:
        norm_seg = _normalize(cur_seg)
        cv2.polylines(img, [norm_seg], isClosed=False, color=0, thickness=1)

    # 返回所有有效点归一化后的坐标（保序）
    norm_points = _normalize(down_points.copy())

    return img.astype(np.float32) / 255.0, norm_points


def generate_path_signature_features(points, max_level=5, target_size=54):
    """
    完整生成 63 通道的路径签名特征图。

    注意：本函数保留用于独立调用，dataset.py 里是内联实现的等价逻辑。
    两者都遵守同一原则：先过滤 -1，再算签名。
    """
    points = np.array(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    down_points = points[points[:, 0] != -1]
    if len(down_points) < 3:
        return np.zeros((target_size, target_size, 63), dtype=np.float32)

    # 计算路径签名（在 CPU 上，避免多进程 CUDA 问题）
    traj_tensor = torch.from_numpy(down_points.astype(np.float32)).unsqueeze(0)
    try:
        sig_tensor = signatory.signature(traj_tensor, max_level).squeeze(0)
    except Exception:
        return np.zeros((target_size, target_size, 63), dtype=np.float32)

    sig_np = sig_tensor.numpy()

    # 归一化点坐标
    min_x, max_x = np.min(down_points[:, 0]), np.max(down_points[:, 0])
    min_y, max_y = np.min(down_points[:, 1]), np.max(down_points[:, 1])
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    scale = min((target_size - 4) / width, (target_size - 4) / height)
    norm_points = down_points.astype(np.float32)
    norm_points[:, 0] = (norm_points[:, 0] - min_x) * scale + 2
    norm_points[:, 1] = (norm_points[:, 1] - min_y) * scale + 2
    norm_points = norm_points.astype(int)

    n_rows = len(norm_points) - 1
    if n_rows <= 0:
        return np.zeros((target_size, target_size, 63), dtype=np.float32)

    # 强制填充/截断到 n_rows × 63
    flat_sig = sig_np.flatten()
    target_len = n_rows * 63
    if len(flat_sig) < target_len:
        flat_sig = np.pad(flat_sig, (0, target_len - len(flat_sig)), 'constant')
    else:
        flat_sig = flat_sig[:target_len]
    sig_np = flat_sig.reshape(n_rows, 63)

    # 构造 63 通道特征图
    sig_maps = np.zeros((target_size, target_size, 63), dtype=np.float32)
    for i in range(1, len(norm_points)):
        if i - 1 < sig_np.shape[0]:
            x, y = norm_points[i]
            if 0 <= x < target_size and 0 <= y < target_size:
                sig_maps[y, x, :] = sig_np[i - 1, :]

    # 直方图均衡化
    for c in range(63):
        channel = sig_maps[:, :, c]
        if np.max(channel) - np.min(channel) > 1e-5:
            norm_ch = ((channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255).astype(np.uint8)
            eq_ch = cv2.equalizeHist(norm_ch)
            sig_maps[:, :, c] = eq_ch.astype(np.float32) / 255.0

    return sig_maps