"""
WPTTDataset：缓存切分结果 + 支持多进程/单进程 + 缓存页码映射 + DropSegment 开关。

关键修复：
- threshold=3（v1 用的 180 在 ×0.1 缩放后的坐标上失效，检测不到任何拐点）
- render_trajectory_to_bitmap 按 -1 分段绘制（见 path_signature.py）
- __getitem__ 里签名计算前过滤 -1（避免跳变点污染积分路径）

双路（静态 + 动态）改造：
- 新增 return_dynamic 开关，打开后 __getitem__ 返回 (静态, 动态, 标签) 三元组
- 动态特征在 **DropSegment 之后** 提取，保证两条路看到的是同一个增强样本
- 修复 signatory 异常分支的两个缺陷：
    (a) 原来提前 return 的是二元组，双路模式下会导致解包崩溃；
    (b) 原来提前 return 时跳过了 self.transform，与正常路径不一致。
  现统一由 _build_static_feature 返回全零特征图并走同一个出口。
"""

import os
import gc
import numpy as np
import cv2
cv2.setNumThreads(0)  # fork 模式下禁用 OpenCV 内部多线程
import torch
import signatory
from torch.utils.data import Dataset
from multiprocessing import Pool

from src.data.loader import load_wptt_page
from src.preprocessing.corner import detect_corners
from src.preprocessing.segmentation import (
    pseudo_segment, estimate_avg_char_height, PEN_UP,
)
from src.features.path_signature import render_trajectory_to_bitmap, generate_path_signature_features
from src.features.dynamic import extract_dynamic_sequence
from src.augmentation.drop_segment import get_segments, apply_drop_segment_from_segments


def _load_one_page(file_path):
    return load_wptt_page(file_path)


#: 项目根目录（src/data/dataset.py → 上溯三级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#: 静态特征编码版本 —— **改动特征语义时必须递增**。
#:
#: 为什么需要它：`separator_aware` / `fix_signature_layout` 两个开关只能表达
#: 「用哪条流水线」，表达不了「同一条流水线内部算法改过」。一旦在不翻开关的
#: 情况下改了特征计算，旧的检查点会被**静默续训**——权重和特征对不上，
#: 准确率莫名其妙地低，而续训守卫（它只比开关）拦不住。
#:
#: v2.2.1  2026-10-18  切分的「宽度」定义修正为论文的字面意思：
#:                     论文说 "When the **width of a character** exceeds the average
#:                     height after adding a segment"，指已收进来的所有点合起来的
#:                     水平跨度。旧实现把每个片段**自身**的宽度累加，片段在水平方向
#:                     重叠时重复计数，导致切得过碎。实测（12 页，真实字符数 251）：
#:                       累加段宽 -> 554 个/页、15 点/字、h/w 0.97、片段中位 2
#:                       包围盒宽 -> 284 个/页、32 点/字、h/w 1.17、片段中位 4
#:                     论文明确 h/w 取 1（[1,1.3] 亦可），包围盒版落在区间内。
#:                     样本总量 2.55M -> 1.17M，单轮耗时回到 ~20 分钟。
#: v2.2.0  2026-10-18  按论文逐项对齐（这一版特征与 v2.1.x **不兼容**）：
#:                     ① 切分阈值改为论文的「先量平均字高再切」（旧值是 page_height/10，
#:                        实测偏大 1.78 倍，伪字符 h/w 只有 0.70；现在 0.96~0.97，
#:                        每页伪字符数 260 → 531，样本总量 1.19M → 2.55M）
#:                     ② 几何增强（平移+旋转+缩放）改到**坐标域**、在算签名之前
#:                        （旧实现作用在渲染好的特征图上，签名数值其实没变，且缺缩放）
#:                     ③ 去掉签名的尺度归一化（论文未做；且论文指出高阶签名更有
#:                        判别力，归一化会抹平这个结构）
#:                     ④ 位图通道改为可配置（use_bitmap），默认 true
#:                     ⑤ DropSegment 去掉「片段数<=3 跳过」的门槛（窄伪字符片段数
#:                        中位只有 2，旧门槛会跳过 89.8% 的样本）
#: v2.1.3  2026-10-17  路径签名改为**尺度归一化**后再计算
#:                     （原先直接用像素坐标，level-5 比 level-1 大 7 个数量级）；
#:                     签名点场做 **4×4 膨胀**，对齐 ConvNeXt stem 的 patchify 尺度
#:                     （否则 stride-4 混叠成噪声，ConvNeXt 会塌缩到 ln(K)）
#: v2.1.0  更早        补上原文 k=0 项，得到 1 位图 + 63 签名（A1）
#: v2.0               原始实现（A0）
STATIC_FEATURE_VERSION = "v2.2.1"


# ----------------------------------------------------------------------
# 切分结果磁盘缓存
#
# 测试集是**固定**的（同一个 metadata、同一套切分参数 → 永远同一批伪字符），
# 但每次建 WPTTDataset 都要重新做三件很贵的事：
#   1. 读取上千个 .wptt 文件
#   2. 逐页做拐点检测（每页 400+ 次 k-cosine 卷积）
#   3. 逐页切分，再对每个伪字符做一次 `get_segments`（又一次拐点检测）
#
# 建好之后把「伪字符坐标 + 片段索引」落盘，下次直接 load。
# 落盘的是**坐标**（不是特征图）：特征图 64x54x54 float32 约 746 KB/样本，
# 测试集 58 万样本要 400+ GB，不可行；而坐标每个伪字符只要几百字节，
# 整个测试集约 140 MB，可行且能全部放进内存。
#
# 缓存键包含 STATIC_FEATURE_VERSION，所以切分/特征算法一变，旧缓存自动失效。
# ----------------------------------------------------------------------

SEG_CACHE_VERSION = "segcache-v1"


def _seg_cache_key(split, metadata_file, n_pages):
    """由（分裂、metadata 内容、页数、特征版本）算缓存文件名。"""
    import hashlib
    h = hashlib.md5()
    for part in (SEG_CACHE_VERSION, STATIC_FEATURE_VERSION, split, str(n_pages)):
        h.update(part.encode('utf-8'))
        h.update(b'\0')
    try:
        with open(metadata_file, 'rb') as f:
            h.update(f.read())
    except Exception:
        h.update(str(metadata_file).encode('utf-8'))
    return h.hexdigest()[:20]


def _save_seg_cache(path, samples, char_points_list, segments_list,
                    page_avg_height):
    """
    落盘切分结果。

    坐标和片段都拼成**一个大数组 + 偏移量**，而不是几百个小数组：
    既省空间（没有 per-array 的对象开销），也让加载后能用切片视图共享内存。
    """
    n = len(char_points_list)
    pt_off = np.zeros(n + 1, dtype=np.int64)
    seg_off = np.zeros(n + 1, dtype=np.int64)
    pts_parts = []
    seg_parts = []
    for i, (pts, segs) in enumerate(zip(char_points_list, segments_list)):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        pts_parts.append(pts)
        pt_off[i + 1] = pt_off[i] + len(pts)
        if segs:
            seg_parts.append(np.asarray(segs, dtype=np.int32).reshape(-1, 2))
        seg_off[i + 1] = seg_off[i] + len(segs)

    points = (np.concatenate(pts_parts, axis=0) if pts_parts
              else np.zeros((0, 2), dtype=np.float32))
    segs_flat = (np.concatenate(seg_parts, axis=0) if seg_parts
                 else np.zeros((0, 2), dtype=np.int32))

    samples_arr = np.array([[p, c] for p, c, _ in samples], dtype=np.int32)
    np.savez(
        path,
        points=points, pt_off=pt_off,
        segs=segs_flat, seg_off=seg_off,
        samples=samples_arr,
        page_avg_height=np.asarray(page_avg_height, dtype=np.float32),
    )


def _load_seg_cache(path):
    """读回切分结果；坐标用切片视图，不再复制。"""
    with np.load(path, allow_pickle=False) as z:
        points = z['points']
        pt_off = z['pt_off']
        segs = z['segs']
        seg_off = z['seg_off']
        samples_arr = z['samples']
        page_avg_height = z['page_avg_height']

    n = len(samples_arr)
    char_points_list = [points[pt_off[i]:pt_off[i + 1]] for i in range(n)]
    segments_list = []
    for i in range(n):
        blk = segs[seg_off[i]:seg_off[i + 1]]
        segments_list.append([(int(a), int(b)) for a, b in blk])
    samples = [(int(p), int(c), None) for p, c in samples_arr]
    return samples, char_points_list, segments_list, list(page_avg_height)


def _resolve_data_path(path):
    """
    解析 metadata 中 `file` 列的路径。

    metadata.csv 里的路径可能是：
      · 绝对路径（在生成它的那台机器上有效，换机器就失效）
      · 相对路径（如 data/raw/WPTT2.0-Train/001-P16.wptt）

    相对路径一律相对**项目根目录**解析，而不是当前工作目录。
    这样无论从哪个目录调用 WPTTDataset，数据都能被找到——
    这是「打包上传、解压即用」的关键。
    """
    text = str(path).replace('\\', os.sep).replace('/', os.sep)
    if os.path.isabs(text):
        return text
    return os.path.join(PROJECT_ROOT, text)


def _segment_one_page(args):
    page_idx, points = args
    corners = detect_corners(points, k=2, threshold=3)
    # 按论文的做法先量平均字高，再切（见 estimate_avg_char_height）
    avg_h = estimate_avg_char_height(points, corners)
    chars = pseudo_segment(points, corners, avg_height=avg_h)
    return page_idx, chars, avg_h


# ----------------------------------------------------------------------
# 路径签名：正确维度处理
# ----------------------------------------------------------------------

def _signature_dim(depth=5, dim=2):
    """level-1..depth 的截断签名维度：2 维轨迹 depth=5 时为 2+4+8+16+32 = 62。"""
    return sum(dim ** k for k in range(1, depth + 1))


def _signature_dim_with_level0(depth=5, dim=2):
    """
    含 k=0 常值项的签名维度 = 63（depth=5, dim=2）。

    DeepWriterID 原文（arXiv:1508.04945）Section 3 明确写道：

        "The dimension of T^k_0 P is 2^k. For algebraic reasons, when k = 0,
         the signature is constant at 1, denoting the original input, while
         k = 1 and k = 2 represent the path displacement and path curvature,
         respectively."
        "... the number of feature maps can be calculated as
         [ sum_{i=0}^{n} 2^i ]"

    n = 5 时 sum_{i=0}^{5} 2^i = 1+2+4+8+16+32 = **63**。
    再加上 1 张位图正好 64 —— 这正是本仓库 in_chans=64 的由来。

    ⚠️ `signatory.signature(x, 5)` 只返回 level 1..5 共 **62** 维，
    **不含 k=0 那一维**，必须手工补上。v2.0 就是漏了这一步，
    却又按 63 去 reshape，才导致签名通道退化。
    """
    return 1 + _signature_dim(depth, dim)


def _prefix_signature_field(down_points, norm_points, depth=5, target_size=54):
    """
    逐点前缀签名特征图（63 通道，含 k=0 常值项）。

    v2.0 的实现把 62 维签名按 63 的步长切开再 reshape，导致只有第 0 行拿到
    真实数据、其余全是 0，63 个通道最终退化成「常数底图 + 黑白极性」。

    这里按原文定义做对：
      · 每个点产生 63 维签名向量 = [k=0 的常值 1] + [level 1..5 的 62 维]；
      · 第 i 个点放的是「从该连续段起点到第 i 个点」这段前缀路径的签名
        （原文: "each sample point along the segment can generate a set of
        signature values with the truncated level n"），
        因此每个通道都是**空间可分辨**的特征图，而不是常数。

    用 Chen 恒等式做增量累加（signatory.signature_combine），
    复杂度 O(n) 而不是 O(n^2)。

    Args:
        down_points: (M, 2) 已过滤掉 -1 的有效点。
        norm_points: (M, 2) 归一化到 [0, target_size) 的整数坐标，与 down_points 对齐。
        depth: 签名截断层数。
        target_size: 特征图边长。

    Returns:
        (target_size, target_size, 63) 的 float32 特征图，未做直方图均衡。

    尺度问题
    --------
    路径签名对坐标尺度敏感：level-n 项的量级约为「路径长度^n」。实测原始坐标下
        level1≈16  level2≈150  level3≈1.2e3  level4≈1.5e4  level5≈1.3e5(峰值 5.2e5)
    即单条路径内跨 7 个数量级。

    早期版本为此加了一步「中心化 + 除以最大边长」。现按论文去掉——论文 §3 指出
    高阶签名更有判别力（"differences between the maps beyond the second level k>=2
    are appreciable ... higher levels may contain information that will contribute
    positively"），而尺度归一化把所有层压到同一量级，正好抹平了这个结构。

    去掉之后仍能训练，靠的是下面这一步 **4×4 膨胀**（见函数末尾）。
    """
    import signatory

    m = len(down_points)
    sig_dim = _signature_dim(depth)              # 62（level 1..5）
    total_dim = 1 + sig_dim                      # 63（含 k=0）
    sig_maps = np.zeros((target_size, target_size, total_dim), dtype=np.float32)
    if m < 2:
        return sig_maps

    # ---- 直接使用原始坐标（论文未做尺度归一化）----
    # 早期版本在这里做了「中心化 + 除以最大边长」，动机是压制 level1..5 之间的
    # 动态范围（实测原样跨 7 个数量级：level1≈16 → level5≈130000）。
    #
    # 但论文 §3 明确说高阶签名更有判别力：
    #   "the maps of the first three iterated integrals seem similar in magnitude
    #    among the writers, while the differences between the maps beyond the
    #    second level k>=2 are appreciable ... higher levels of the path-signature
    #    may contain information that will contribute positively"
    # 尺度归一化把所有层压到同一量级，等于抹平了「高阶更重要」这个结构。
    #
    # 消融实测（24 类、2 seeds）：只要做了 4×4 膨胀，不做尺度归一化也能让
    # ConvNeXt 学会（12 步），甚至比做了归一化更快（16 步）。所以这里去掉。
    pts_unit = down_points.astype(np.float32)

    accum = None

    for i in range(1, m):
        p_prev = torch.from_numpy(pts_unit[i - 1:i])
        p_cur = torch.from_numpy(pts_unit[i:i + 1])
        step = torch.cat([p_prev, p_cur], dim=0).unsqueeze(0)   # (1, 2, 2)

        # 判断这一步是否与上一步连续：归一化坐标不相邻则视为断裂，重新起算
        x, y = norm_points[i]
        px, py = norm_points[i - 1]
        contiguous = (abs(int(x) - int(px)) <= 1) and (abs(int(y) - int(py)) <= 1)

        try:
            step_sig = signatory.signature(step, depth).squeeze(0)
        except Exception:
            accum = None
            continue

        if accum is None or not contiguous:
            accum = step_sig
        else:
            try:
                accum = signatory.signature_combine(
                    accum.unsqueeze(0), step_sig.unsqueeze(0), 2).squeeze(0)
            except Exception:
                accum = step_sig

        if 0 <= x < target_size and 0 <= y < target_size:
            v = accum.detach().cpu().numpy()
            if len(v) == sig_dim:
                # k=0 项恒为 1（原文: "the signature is constant at 1"）
                sig_maps[y, x, 0] = 1.0
                sig_maps[y, x, 1:] = v

    # ---- 把稀疏点场加密到 ConvNeXt stem 的尺度 ----
    # ConvNeXt V2 的 stem 是 Conv2d(kernel=4, stride=4)：一个 4×4 块只输出一个值。
    # 而前缀签名本质是**逐点**的稀疏场（每个通道约 50 个非零像素 / 2916）。
    # 4×4 平均会把单点信号稀释 16 倍，和相邻的背景混在一起变成噪声 ——
    # 实测 ConvNeXt 在这种输入上连 24 个样本都过拟合不了（loss 卡在 ln(24)=3.178，
    # 即塌缩成均匀输出），而 DCNN 用 3×3 stride-1 没有这个问题。
    #
    # 用相同的 4×4 结构元做膨胀，让每个签名点至少覆盖一个完整的 stem patch。
    # 实测（24 类、3 个随机种子）：
    #     不膨胀        1/3 成功
    #     3×3 膨胀      2/3 成功
    #     4×4 膨胀      3/3 成功（16~37 步，比 A0 的 74~173 步还快）
    if total_dim > 0:
        kernel = np.ones((4, 4), np.uint8)
        for c in range(total_dim):
            sig_maps[:, :, c] = cv2.dilate(sig_maps[:, :, c], kernel)

    return sig_maps


def _equalize_channels(sig_maps):
    """对每个通道做 min-max 归一化 + 直方图均衡（保持原实现的处理方式）。"""
    result = sig_maps.copy()
    for c in range(result.shape[-1]):
        channel = result[:, :, c]
        if np.max(channel) - np.min(channel) > 1e-5:
            norm_ch = ((channel - np.min(channel)) /
                       (np.max(channel) - np.min(channel)) * 255).astype(np.uint8)
            eq_ch = cv2.equalizeHist(norm_ch)
            result[:, :, c] = eq_ch.astype(np.float32) / 255.0
    return result


class WPTTDataset(Dataset):
    def __init__(self, metadata_file, split='Train', transform=None,
                 page_workers=8, segment_workers=8, apply_drop=True,
                 return_dynamic=False, seq_len=200, dynamic_speed_clip=10.0,
                 augment_points=False, use_bitmap=True, drop_max_segments=1,
                 cache_dir='data/cache'):
        """
        Args:
            metadata_file: metadata.csv 路径
            split: 'Train' 或 'Test'
            transform: 图像变换（如 _pad_to_96），只作用于静态特征图
            page_workers: 页面加载的进程数，0 表示单进程
            segment_workers: 伪字符切分的进程数，0 表示单进程
            apply_drop: 是否在 __getitem__ 里做 DropSegment
                - True：训练集、20-test 评估使用
                - False：训练每轮评估使用（保证测试内容一致）
            return_dynamic: 是否额外返回伪时序动态特征序列
                - False：返回 (静态, 标签)
                - True ：返回 (静态, 动态, 标签)，动态形状为 (seq_len, 2)
            seq_len: 动态序列长度
            dynamic_speed_clip: 归一化伪速度的截断上限
            augment_points: 是否在**坐标域**做仿射增强（平移+旋转+缩放）。
                论文要求在「算签名特征图之前」做，见 `_augment_points`。
                训练置 True，评估必须 False（否则测试集不再确定）。
            use_bitmap: 是否把位图作为第 0 通道（默认 True = 论文的
                「签名与位图一起」；False = 只有 63 张签名图）。
            drop_max_segments: DropSegment 的 **m**（论文 §3 的固定上限）。
                每次随机删 r ~ Uniform{0..m} 段，**不是百分比**。
                默认 1 = 最保守读法，实测平均丢弃 19.8% 的墨迹。
            cache_dir: 切分结果的磁盘缓存目录。测试集是固定的，第一次建好后
                落盘（约 140 MB），以后直接加载，跳过读盘/拐点检测/切分。
                传 None 关闭缓存。

        切分阈值按论文的做法**先量再切**（`estimate_avg_char_height`），
        每个页面的平均字高在构建时算一次并缓存到 `page_avg_height`。
        """
        import pandas as pd
        self.df = pd.read_csv(metadata_file, encoding='utf-8-sig')
        self.df['writer_id'] = self.df['writer_id'].astype(str)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.writer_ids = sorted(self.df['writer_id'].unique())
        self.label_map = {wid: idx for idx, wid in enumerate(self.writer_ids)}
        self.num_classes = len(self.writer_ids)
        self.transform = transform
        self.apply_drop = apply_drop
        self.return_dynamic = return_dynamic
        self.seq_len = seq_len
        self.dynamic_speed_clip = dynamic_speed_clip
        self.augment_points = augment_points
        self.use_bitmap = use_bitmap
        self.drop_max_segments = drop_max_segments
        # 伪字符二次切分裂成多块的次数（罕见，见 __getitem__；多进程下各进程独立计数）
        self._resplit_count = 0

        # ========== 页面加载 / 伪字符切分（带磁盘缓存）==========
        # 切分结果与「训练/评估」无关，只与 metadata + 切分算法有关。
        # 缓存命中时跳过：读 .wptt、逐页拐点检测、逐字符 get_segments —— 三步都很贵。
        cache_path = None
        if cache_dir:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(
                    cache_dir,
                    f'{split}_{_seg_cache_key(split, metadata_file, len(self.df))}.npz')
            except OSError as exc:
                # 缓存目录不可写（只读挂载 / 权限不足）时**不能**让整个建集失败，
                # 缓存只是加速手段，没有它也能正常跑，只是慢一点。
                print(f"⚠️  缓存目录不可用，本次跳过切分缓存：{exc}")
                cache_path = None

        loaded = False
        if cache_path and os.path.exists(cache_path):
            try:
                samples, char_points_list, segments_list, page_avg_height = \
                    _load_seg_cache(cache_path)
                self.samples = samples
                self.char_points_list = char_points_list
                self.segments_list = segments_list
                self.page_avg_height = page_avg_height
                loaded = True
                size_mb = os.path.getsize(cache_path) / 1024 / 1024
                print(f"⚡ 切分缓存命中：{cache_path} （{size_mb:.0f} MB，"
                      f"{len(samples)} 个伪字符，跳过读盘与切分）")
            except Exception as exc:                     # noqa: BLE001
                print(f"⚠️  切分缓存损坏，重新构建：{type(exc).__name__}: {exc}")
                loaded = False

        if not loaded:
            # 相对路径按项目根目录解析，保证换机器/换工作目录也能找到数据
            file_list = [_resolve_data_path(f) for f in self.df['file'].tolist()]
            if page_workers > 0:
                print(f"正在加载 {split} 集的所有页面轨迹（{page_workers} 进程）...")
                with Pool(processes=page_workers) as pool:
                    self.pages = pool.map(_load_one_page, file_list)
            else:
                print(f"正在加载 {split} 集的所有页面轨迹（单进程）...")
                self.pages = [_load_one_page(f) for f in file_list]
            print(f"✅ 加载了 {len(self.pages)} 个页面。")

            # ========== 伪字符切分 ==========
            page_args = [(i, points) for i, points in enumerate(self.pages)]
            if segment_workers > 0:
                print(f"正在预切分伪字符并建立索引（{segment_workers} 进程）...")
                with Pool(processes=segment_workers) as pool:
                    results = pool.map(_segment_one_page, page_args)
            else:
                print(f"正在预切分伪字符并建立索引（单进程）...")
                results = [_segment_one_page(args) for args in page_args]

            # ========== 缓存切分后的字符坐标 ==========
            self.samples = []
            self.char_points_list = []
            # 每页的切分阈值（字符平均高度），__getitem__ 重切单字时必须沿用，
            # 不能对单个字形重新估计
            self.page_avg_height = [None] * len(self.df)
            _wid_by_page = self.df['writer_id'].tolist()
            for page_idx, chars, avg_h in results:
                self.page_avg_height[page_idx] = avg_h
                if len(chars) > 0:
                    writer_id = _wid_by_page[page_idx]
                    for char_idx in range(len(chars)):
                        self.samples.append((page_idx, char_idx, writer_id))
                        pts = chars[char_idx]
                        if pts.ndim == 1:
                            pts = pts.reshape(-1, 2)
                        self.char_points_list.append(pts)
            print(f"✅ 共生成 {len(self.samples)} 个伪字符样本，坐标已缓存。")

            # ========== 释放页面级轨迹 ==========
            self.pages = None
            gc.collect()
            print(f"🧹 已释放页面级轨迹缓存，节省约 5-10GB 内存。")

            # ========== 缓存 DropSegment 片段索引 ==========
            print(f"正在缓存 DropSegment 片段索引（{len(self.char_points_list)} 个样本）...")
            self.segments_list = []
            for char_pts in self.char_points_list:
                self.segments_list.append(get_segments(char_pts))
            print(f"✅ 已缓存 {len(self.segments_list)} 个样本的 DropSegment 片段索引。")

            if cache_path:
                try:
                    _save_seg_cache(cache_path, self.samples, self.char_points_list,
                                    self.segments_list, self.page_avg_height)
                    print(f"💾 切分结果已缓存：{cache_path} "
                          f"（{os.path.getsize(cache_path)/1024/1024:.0f} MB）")
                except Exception as exc:                 # noqa: BLE001
                    print(f"⚠️  写缓存失败（不影响本次运行）：{type(exc).__name__}: {exc}")

        # 样本里的 writer_id 从 df 重新补全（缓存不存它，避免冗余）。
        # 用 list 预取 + 整数索引，避免逐样本 df.iloc[...]（58 万样本时慢到秒级）。
        _wid_by_page = self.df['writer_id'].tolist()
        _name_by_page = self.df['page'].tolist()
        self.samples = [(p, c, _wid_by_page[p]) for p, c, _ in self.samples]
        self.page_names_for_samples = [_name_by_page[p] for p, _, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    # 静态特征构造
    #
    # 数值与原内联实现逐位一致，仅把「提前 return」改为「返回全零特征图」，
    # 使异常路径也能走统一的出口（transform + 三元组）。
    # ------------------------------------------------------------------

def _augment_points(points, max_rot_deg=5.0, max_trans=0.05,
                    scale_range=(0.9, 1.1)):
    """
    对**笔迹坐标**做仿射增强（旋转 + 平移 + 缩放）。

    论文原文（§5 "Investigation of Path-Signature Feature"）：
        "**During the training stage, before the extraction of path-signature
         feature maps**, we adopted a random mix of affine transformations i.e.,
         translation, rotation, scaling as the elastic distortion to achieve
         further data augmentation and generalized applicability."

    两个要点：
      1. 变换是作用在**坐标**上的，在算签名特征图**之前**。
         签名是路径的几何泛函，所以变换后重算的签名才是「变换后路径的签名」。
         早期实现是在渲染好的 64 通道特征图上做 RandomRotation/RandomAffine，
         签名数值其实没变，只是同一张图被转过头了——不符合论文。
      2. 三个分量都要有。早期实现只有旋转和平移，**缺缩放**。

    注意：`render_trajectory_to_bitmap` 内部按包围盒做**统一缩放**适配 54×54，
    所以平移和等比缩放对位图基本无效（会被吃掉），但**对签名有效**——因为签名
    直接在这些坐标上计算。这正是「变换必须在坐标域」的价值所在。

    Args:
        points: (N, 2)，可能含 (-1, 0) 抬笔分隔符行（这些行原样保留）。

    Returns:
        变换后的 (N, 2) float32。

    随机数用**模块级** `np.random`，不用私有 Generator：DataLoader 的
    `worker_init_fn` 会按 worker_id 重设 `np.random` 的种子，各 worker 因此
    互相独立；而私有 Generator 在 fork 时会被复制成同一状态，导致各 worker
    产生完全相同的增强。
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    valid = pts[:, 0] != PEN_UP
    if valid.sum() < 2:
        return pts

    xy = pts[valid]
    center = xy.mean(axis=0, keepdims=True)

    theta = np.deg2rad(np.random.uniform(-max_rot_deg, max_rot_deg))
    s = np.random.uniform(*scale_range)
    c, sn = np.cos(theta), np.sin(theta)
    R = np.array([[c, -sn], [sn, c]], dtype=np.float32) * s

    extent = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    if extent < 1e-6:
        extent = 1.0
    shift = (np.random.uniform(-max_trans, max_trans, size=(1, 2))
             .astype(np.float32) * extent)

    pts[valid] = (xy - center) @ R.T + center + shift
    return pts


def _build_static_feature(final_points, down_points, use_bitmap=True):
    """
    构造静态特征图：`use_bitmap=True` 时 64 通道（1 位图 + 63 签名），
    否则 63 通道（只有签名）——后者对应论文里 M 的另一种读法。

    通道布局
    --------
    论文对输入通道数有两处互相矛盾的表述：
      · "M denotes the number of input channels, which is equal to the number
         of signature feature maps."                            → M = 63
      · "the path-signature feature **together with Bitmap** can greatly enhance
         the network representation"                             → 位图也进网络
      · Fig.4 的消融曲线是 Bitmap、Bitmap+Sign.1 … Bitmap+Sign.5 → 位图始终在
    按后两处，提出方法 = 位图 + 签名 = 1 + 63 = 64。这里做成开关，两种都能跑。

    位图通道的用途
    --------------
    它在论文里同时是**对照方法**（"The baseline method denoted by Bitmap ...
    render the online handwritten characters as offline bitmaps to train the DCNN"），
    用来证明路径签名相对纯位图的增益。

    任何环节失败都返回全零特征图，不抛异常。
    """
    feat = np.zeros((54, 54, 64), dtype=np.float32)
    sig_slot = 1 if use_bitmap else 0

    # 签名
    _bitmap, norm_points = render_trajectory_to_bitmap(final_points)
    if len(down_points) < 3:
        return feat
    try:
        sig_maps = _prefix_signature_field(down_points, norm_points)
    except Exception:
        return feat
    sig_maps = _equalize_channels(sig_maps)          # (54, 54, 63)

    # 位图（内部按 -1 分段绘制）
    if use_bitmap:
        bitmap, _ = render_trajectory_to_bitmap(final_points)
        feat[:, :, 0] = bitmap

    n_sig = min(sig_maps.shape[-1], 63 - sig_slot if not use_bitmap else 63)
    feat[:, :, sig_slot:sig_slot + n_sig] = sig_maps[:, :, :n_sig]
    return feat


# ----------------------------------------------------------------------
# 以下两个方法在模块级定义（便于单独测试），文件末尾再挂回 WPTTDataset。
# 放在模块级是因为它们不依赖 self；挂回去保持
# `self.__getitem__` / `WPTTDataset._build_static_feature(...)` 两种调用都可用。
# ----------------------------------------------------------------------

def __getitem__(self, idx):
        page_idx, char_idx, writer_id = self.samples[idx]
        char_points = self.char_points_list[idx]

        # ========== DropSegment 开关 ==========
        if self.apply_drop:
            drop_points = apply_drop_segment_from_segments(
                char_points, self.segments_list[idx],
                max_remove=self.drop_max_segments,
            )
        else:
            drop_points = char_points

        # ========== 几何增强（必须在特征提取之前，见 _augment_points）==========
        # 论文：仿射变换（平移 + 旋转 + 缩放）作用于**坐标**，然后再算签名特征图。
        if self.augment_points:
            drop_points = _augment_points(drop_points)

        # ========== 动态特征（伪时序） ==========
        # 必须基于 drop_points：DropSegment 每次随机删片段，动态特征也应基于
        # 删后的轨迹提取，否则两路看到的不是同一个增强样本。
        # drop_points 中保留了 -1 分段标记，extract_dynamic_sequence 会在这些
        # 位置切分，避免跨段产生量级 ~100 的虚假速度尖峰。
        dynamic_tensor = None
        if self.return_dynamic:
            dynamic_seq = extract_dynamic_sequence(
                drop_points,
                seq_len=self.seq_len,
                speed_clip=self.dynamic_speed_clip,
            )
            dynamic_tensor = torch.from_numpy(dynamic_seq).float()

        # ========== 静态特征 ==========
        new_corners = detect_corners(drop_points, k=2, threshold=3)
        # 用**该页**估计出的平均字高做阈值。绝不能在这里再自动估一次——
        # 输入已经是单个伪字符，对它重估会把本来完整的一个字切开。
        chars_drop = pseudo_segment(drop_points, new_corners,
                                    avg_height=self.page_avg_height[page_idx])

        # `char_points_list` 里的每个元素本来就是 `pseudo_segment` 产出的一个完整
        # 伪字符，对它再切一次通常还是它自己。但**不总是**：宽于 50 像素的字符
        # 会被 `avg_height` 的兜底值切开（实测 Test 约 0.02%）。
        #
        # 早期版本写的是 `final_points = chars_drop[0]` —— 多出来的块被**静默丢弃**。
        # 那个 bug 在「抬笔分隔符被抹平」时会大规模爆发：拼接点坐标突变 →
        # detect_corners 找到假角点 → 二次切分裂开。实测旧实现下 Train 24.9% /
        # Test 22.3% 的样本被截断，平均丢弃 21.9% 的墨迹，最坏丢 95%，
        # 而参数量、张量形状、样本数全都看不出异常。
        #
        # 现在改为**拼接所有块**（块之间插分隔符），宁可多一个分隔符也不丢数据。
        # 单块时等价于原行为；`self._resplit_count` 统计发生次数，训练结束可查。
        if len(chars_drop) > 1:
            parts = []
            for k, piece in enumerate(chars_drop):
                if piece.ndim == 1:
                    piece = piece.reshape(-1, 2)
                if k:
                    parts.append(np.array([[PEN_UP, 0.0]], dtype=piece.dtype))
                parts.append(piece)
            final_points = np.vstack(parts)
            self._resplit_count += 1
            if self._resplit_count == 1:
                print(f"ℹ️  伪字符二次切分：索引 {idx} 裂成 {len(chars_drop)} 块"
                      f"（各 {[len(x) for x in chars_drop]} 点），已拼接而非丢弃。"
                      f"该情形罕见，不影响整体。")
        elif len(chars_drop) == 1:
            final_points = chars_drop[0]
            if final_points.ndim == 1:
                final_points = final_points.reshape(-1, 2)
        else:
            final_points = None

        feat = np.zeros((54, 54, 64), dtype=np.float32)
        if final_points is not None and len(final_points) > 0:
            # 过滤掉 -1（抬笔分隔符），用于签名计算
            down_points = final_points[final_points[:, 0] != -1.0]

            if len(down_points) >= 3:
                feat = _build_static_feature(
                    final_points, down_points, use_bitmap=self.use_bitmap)

        feat_tensor = torch.from_numpy(feat).permute(2, 0, 1).float()
        if self.transform:
            feat_tensor = self.transform(feat_tensor)

        label = self.label_map[writer_id]

        if self.return_dynamic:
            return feat_tensor, dynamic_tensor, label
        return feat_tensor, label


# ----------------------------------------------------------------------
# 把上面两个模块级函数挂回类上
# ----------------------------------------------------------------------
# `_build_static_feature` 用 staticmethod 包装，这样
# `WPTTDataset._build_static_feature(...)`（诊断/冒烟脚本在用）和
# `self._build_static_feature(...)` 都能工作。
WPTTDataset._build_static_feature = staticmethod(_build_static_feature)
WPTTDataset.__getitem__ = __getitem__
