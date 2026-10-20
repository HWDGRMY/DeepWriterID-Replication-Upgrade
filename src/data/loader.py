import os
import struct
import numpy as np


def load_wptt_page(file_path):
    """
    基于 CASIA 官方 WPTT 结构解析单页文本轨迹。
    严格遵循 CSDN 文章文件头结构解析。
    """
    all_points = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'rb') as f:
        # ================= 1. 解析 WPTT 文件头 =================
        # 头部总长度 (4B, 小端序 uint32)
        header_len = struct.unpack('<I', f.read(4))[0]

        # Format code (8B, 固定 'WPTT')
        fmt = f.read(8)
        if not fmt.startswith(b'WPTT'):
            raise ValueError(f"文件格式错误，期望 WPTT，实际得到 {fmt}")

        # Illustration (可变长字符串，以 \0 结尾)
        illustration = b''
        while True:
            c = f.read(1)
            if not c or c == b'\0':
                break
            illustration += c

        # Code type (20B, 固定 'GB')
        code_type = f.read(20)
        # Code length (2B, 短整型, 固定 2)
        code_len = struct.unpack('<H', f.read(2))[0]
        # Data type (20B, 固定 'short')
        data_type = f.read(20)

        # Sample length (4B, 文件总字节数)
        sample_len = struct.unpack('<I', f.read(4))[0]
        # Page index (4B, 页码)
        page_idx = struct.unpack('<I', f.read(4))[0]
        # Stroke number (4B, 当前页所有笔画总数)
        stroke_num = struct.unpack('<I', f.read(4))[0]

        # ================= 2. 读取全部笔画 =================
        all_strokes = []
        for _ in range(stroke_num):
            # 单笔画包含的点数 (2B)
            pt_count = struct.unpack('<H', f.read(2))[0]
            stroke_pts = []
            for _ in range(pt_count):
                # 坐标 X (2B, 有符号短整型) -> 实际坐标需除以 10
                x = struct.unpack('<h', f.read(2))[0] / 10.0
                # 坐标 Y (2B, 有符号短整型)
                y = struct.unpack('<h', f.read(2))[0] / 10.0
                stroke_pts.append((x, y))
            all_strokes.append(stroke_pts)

        # ================= 3. 读取文本行与字符编码 =================
        # 文本行总数 (2B)
        line_num = struct.unpack('<H', f.read(2))[0]

        for _ in range(line_num):
            # 该行的笔画数量 (2B)
            line_stroke_num = struct.unpack('<H', f.read(2))[0]
            # 该行包含的笔画索引数组 (2B * line_stroke_num)
            line_stroke_idx = []
            for _ in range(line_stroke_num):
                idx = struct.unpack('<H', f.read(2))[0]
                line_stroke_idx.append(idx)

            # 该行的字符数量 (2B)
            line_char_num = struct.unpack('<H', f.read(2))[0]
            # 字符 GB 编码 (2B * line_char_num)
            line_chars = []
            for _ in range(line_char_num):
                # 固定 2 字节 GB2312 编码
                tag_code = f.read(code_len)
                # 用 GBK 解码（GB2312 的超集），处理乱码
                char = tag_code.decode('gbk', errors='ignore')
                line_chars.append(char)

    # ================= 4. 拼装为论文所需的轨迹序列 =================
    for stroke in all_strokes:
        if not stroke:
            continue
        for pt in stroke:
            # 过滤掉坐标原点(0,0)附近 2.0 以内的孤立噪点
            if abs(pt[0]) <= 2.0 and abs(pt[1]) <= 2.0:
                continue
            all_points.append(pt)
        all_points.append((-1.0, 0.0))

    return np.array(all_points, dtype=np.float32)