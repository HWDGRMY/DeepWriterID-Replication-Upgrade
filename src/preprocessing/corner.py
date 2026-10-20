import numpy as np

def detect_corners(points, k=2, threshold=3.0):
    corners_idx = []
    n = len(points)
    if n < 2 * k + 1:
        return corners_idx

    bending_values = np.zeros(n)
    for i in range(k, n - k):
        x_i, y_i = points[i]
        x_f, y_f = points[i + k]
        x_b, y_b = points[i - k]
        beta = max(abs(x_f + x_b - 2 * x_i), abs(y_f + y_b - 2 * y_i)) / (2 * k)
        bending_values[i] = beta

    for i in range(k, n - k):
        if bending_values[i] > threshold and bending_values[i] > max(bending_values[i-1], bending_values[i+1]):
            corners_idx.append(i)

    return corners_idx