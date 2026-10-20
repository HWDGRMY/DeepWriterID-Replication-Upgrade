import matplotlib.pyplot as plt
import os
import numpy as np


def draw_trajectory_and_corners(points, corners_idx, save_path):
    """画出轨迹和拐点红点（清爽显示版）"""
    plt.figure(figsize=(12, 7))

    down_points = points[points[:, 0] != -1]

    # 画蓝色轨迹（线宽 0.8）
    plt.plot(down_points[:, 0], -down_points[:, 1], 'b-', linewidth=0.8, label='Trajectory')

    # 画红色拐点（恢复为中等大小，去掉粗黑边）
    if len(corners_idx) > 0:
        corner_points = points[corners_idx]
        plt.scatter(corner_points[:, 0], -corner_points[:, 1],
                    c='red', s=40, marker='o', zorder=10, label='Corners')

    plt.axis('equal')
    plt.legend()
    plt.title('Corner Detection Test')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 调试图片已保存至: {save_path}")