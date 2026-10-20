# 伪时序双路融合 —— 实现与云端运行指南

> 本文档对应方案文档《伪时序双路融合：完整技术方案》的代码落地版本。
> 静态流水线提供 **A0（v2.0 原样，逐位可复现 96.17%）** 与 **A1（修正版）**
> 两个可切换模式，因此消融实验可以同时回答「动态分支有没有用」和
> 「静态流水线原本有没有缺陷」两个问题。

---

## 一、先说五个必须知道的实现事实

方案文档里的伪代码直接落地会踩两个坑；审计静态流水线时又发现三个既有缺陷。
**这五条是论文「方法有效性」论证的核心，建议写进第 3.1 节与局限章节。**

### 1.1 必须先在 `-1` 处分段，再算差分（否则速度特征被彻底淹没）

`drop_points`（DropSegment 增广后的轨迹）里混有 `(-1, 0)` 分段标记。
若直接对整条轨迹求 `np.diff`，`(-1, 0)` 与前一个真实点之间会产生一个
量级 ~100–1000 的虚假「速度」。它会是整条序列的最大值，把真实的速度
变化完全压平。

实测（`scripts/verify_dynamic.py` 第 2 项，合成轨迹）：

| 实现 | 归一化伪速度最大值 |
| :--- | ---: |
| 朴素实现（整条差分） | **61.19** |
| 本实现（分段差分） | **1.00** |

### 1.2 曲率必须做角度环绕修正（否则曲率突破 π 的物理上界）

方向角 `θ = arctan2(dy, dx)` 值域是 `(-π, π]`。当真实转向很小但恰好跨过
`±π` 分支切割时（例如 179° → -179°，真实转向只有 2°），
`np.abs(np.diff(angles))` 会得到 **358° ≈ 6.25 rad** 的虚假曲率。
修复方式是 `(dθ + π) % 2π - π`。

实测（真实 `.wptt` 数据，60 个伪字符）：

| 实现 | 曲率最大值（均值） | 是否物理合法 |
| :--- | ---: | :--- |
| 朴素实现 | **5.05 rad** | ❌ 超过 π=3.14，物理上不可能 |
| 本实现 | **2.71 rad** | ✅ 落在 [0, π] |

> 论文可用的一句话：*朴素实现的曲率上界被突破 60%，说明其数值并非真实转向角，
> 而是分支切割伪影。*

### 1.3 必须按段比例重采样，而不是整条插值

把整条序列直接插值到固定长度，插值核会跨越分段边界，在边界附近制造
并不存在的平滑过渡。本实现按各段长度比例分配时间槽（最大余数法保证槽数
总和恰为 `seq_len`），段内独立插值后再拼接。

---

### 以下三条是**既有静态流水线**的缺陷（不是新增代码引入的）

这三条不影响「代码能跑出 96.17%」，但影响「96.17% 是怎么来的」这一表述。
一条命令给出全部量化证据：

```bash
python scripts/diagnose_static_features.py
```

### 1.4 抬笔分隔符被吞掉，位图里出现跨笔直线

`pseudo_segment` 用 `seg[seg[:, 0] != -1]` 剥掉抬笔标记，再
`np.vstack(current_char)` 首尾相接。于是：

* `render_trajectory_to_bitmap` 里那段「**关键修复：按 -1 分段绘制，
  避免多笔画被一条直线连接**」是**死代码**——`is_pen_up` 全为 False，
  `if pt[0] == -1` 分支永不执行，最后只调用一次 `cv2.polylines`，
  整条轨迹被画成一条折线。
* `signatory.signature(down_points, 5)` 的积分路径同样跨越了抬笔间隙。

8 页实测：

| 指标 | 数值 |
| :--- | ---: |
| 原始页面抬笔标记 | 5439 |
| 含抬笔标记的拐点片段 | 10859 / 19321（56.2%） |
| **存在 >10× 典型点间距跳变的伪字符** | **408 / 1910（21.4%）** |
| 该跳变 / 典型点间距 | 中位数 12.3×，最大 552.3× |

**修复**：`data.separator_aware: true`。

### 1.5 路径签名维度错位：62 被当成 63 用

level-1..5 的截断签名对 **2 维**轨迹的维度是 `2+4+8+16+32 = 62`，不是 63。
v2.0 按 `n_rows * 63` 补齐再 reshape，于是只有第 0 行拿到真实数据，
其余行全是 0；随后 `sig_maps[y, x, :] = sig_np[i-1, :]` 把这**一行**
写到「第 2 个归一化点」这**一个像素**上。

实测（8 页 / 80 样本）：

| 指标 | 位图通道 | 签名通道（1..63） |
| :--- | ---: | ---: |
| 通道内空间标准差（中位数） | **0.238** | **0.017** |
| 样本间标准差（中位数） | 0.187 | 0.453 |
| 单张图内接近常数（σ<0.05）的通道数 | —— | **63 / 63** |

也就是说，63 个签名通道在单张图内几乎是**常数底图**，只有黑白极性随样本
翻转——它编码的是 62 个签名分量的**符号位**（约 62 bit/样本的全局形状信息），
而不是设计意图中的「54×54 空间可分辨签名场」。

**修复**：`data.fix_signature_layout: true`。按 62 维正确处理，并用
Chen 恒等式（`signatory.signature_combine`）做 **O(n)** 的逐点前缀签名累加，
使签名通道成为真正的空间特征图。修复后：通道平均空间标准差
**0.018 → 0.040**，17/63 个通道具备明显空间结构。

> 末尾补 1 个零通道凑满 64，因此 `in_chans=64` 与既有权重
> （含 `convnext_best_9617.pth`）**完全兼容**，热启动不受影响。

### 1.6 动态特征的正确性保证

`scripts/verify_dynamic.py` 提供 28 项断言，含匀速直线（速度恒为 1、曲率为 0）、
跨段无尖峰、角度环绕、槽数守恒、退化输入、真实数据端到端等。**上传云端前先跑一次。**


---

## 二、文件清单

### 新增

| 文件 | 作用 |
| :--- | :--- |
| `src/features/dynamic.py` | 伪时序特征提取（分段感知 + 环绕修正 + 比例重采样） |
| `src/models/dynamic_branch.py` | 动态分支：Transformer Encoder + 注意力池化 |
| `src/models/dual_stream.py` | 双路融合模型（static / dynamic / dual 三模式） |
| `src/training/trainer_dual.py` | 三阶段训练器（含断点续训、消融模式） |
| `scripts/verify_dynamic.py` | 动态特征正确性验证（28 项，仅需 numpy） |
| `scripts/diagnose_static_features.py` | **静态流水线体检**：量化 1.4 / 1.5 两个缺陷 |
| `scripts/run_ablation.sh` | **三组实验一键运行**（固化顺序与依赖） |
| `scripts/dry_run_ablation.py` | **三组实验本地演练**（极小子集走完整流程） |
| `src/models/dcnn_backbone.py` | **原论文 DCNN**（arXiv:1508.04945），权重键名与原复刻仓库一致 |
| `src/training/trainer_multi.py` | **共享管道多模型训练器**：一份内存数据同时喂多个模型 |
| `scripts/train_multi.py` | 多模型训练入口 |
| `configs/multi_arch.yaml` | 多模型训练配置 |
| `scripts/smoke_test_dual.py` | 全套件冒烟测试（不训练到收敛，分钟级） |
| `scripts/train_dual.py` | 训练入口（三种模式共用） |
| `scripts/analyze_complementarity.py` | CCA + 误差四象限 + 配对 t 检验 + t-SNE |
| `configs/dual_stream.yaml` | 主配置 |

### 修改（向后兼容，默认行为不变）

| 文件 | 改动 |
| :--- | :--- |
| `src/preprocessing/segmentation.py` | 新增 `keep_separators`（默认 `False` = 与原实现逐位相同），内部改为索引驱动以保留边界信息 |
| `src/data/dataset.py` | 新增 `return_dynamic` / `separator_aware` / `fix_signature_layout`；修复 signatory 异常分支的元组长度与 transform 缺失 |
| `src/evaluation/evaluator.py` | 兼容 2/3 元组；新增 `extract_test_features` 逐样本导出；多进程上下文改为平台自适应 |

**静态路径默认逐位未改动**：`_build_static_feature` 是从原 `__getitem__` 内联代码原样抽出的，
数值逐位一致，冒烟测试第 3 项会断言 `mode='static'` 与 `ConvNeXtBackbone`
logits **逐元素完全相同**（最大差 0.000e+00）；第 11 项还会断言
`separator_aware=False` 时的 `pseudo_segment` 与重构前的原始算法**逐位相同**。

同时修复了原 `__getitem__` 的一处隐患：`signatory` 抛异常时原来提前 `return`
的是二元组且**跳过了 `self.transform`**，与正常路径不一致。现在统一走全零特征图出口。

---

## 三点五、架构对照：原论文 DCNN vs ConvNeXt V2（共享管道）

### 为什么需要它

论文（arXiv:1508.04945）用的是自研 DCNN；本仓库升级到 ConvNeXt V2。
要论证「升级有效」，必须让两个模型**走完全同一条数据管道**，否则差异里
混进了特征提取、增强、评估协议的差别。

### 一份数据同时喂两个模型

复刻仓库 README 明确写着「GPU 长期处于饥饿状态」——瓶颈是 **CPU 数据管道**
（signatory 五次签名 + 63 次 equalizeHist + 两个几何变换），不是 GPU。
所以开两个进程只是让它们抢同一批 CPU 核，总时间不会减少。

`trainer_multi.py` 改为：

```
一个 WPTTDataset（内存里只有一份）
          ↓
一个 DataLoader（特征提取每个 batch 只做一次）
          ↓
   ┌──────┴──────┐
batch  →  模型A  → loss → backward → step
   └──────→  模型B  → loss → backward → step
```

三个收益：
1. 特征提取**每个 batch 只算一次**，省掉接近一半的 CPU 开销；
2. 两个模型看到的**增强样本完全一致**，架构对比更严格；
3. 显存峰值 ≈ `max(各模型激活)` 而非求和——每个模型 backward 完立刻释放计算图。

### 一步到位：单独训练 / 一起训练（同一个入口）

```bash
python scripts/train_multi.py --list            # 看看配置里有哪些模型
python scripts/train_multi.py                   # 一起训练（全部 runs，共享管道）
python scripts/train_multi.py --only dcnn       # 只单独训练 dcnn
python scripts/train_multi.py --runs convnext   # 只训练列出的几个
python scripts/train_multi.py --set epochs=30
python scripts/train_multi.py --run_override dcnn lr=0.0003
```

`--only` / `--runs` 会自动给组合断点与日志**加后缀**（如 `multi_arch_dcnn`），
所以「单独训练」与「一起训练」的产物不会互相覆盖，可以混着用。

每个 run 也支持**多阶段**（双路融合那条线），给它写 `stages` 即可：

```yaml
- name: dual
  arch: convnext
  mode: dual
  stages:
    - {name: stage1_head_warmup,    epochs: 5,  lr: 1.0e-3, freeze_static: true,  static_lr_scale: 0.0}
    - {name: stage2_head_converge,  epochs: 20, lr: 5.0e-4, freeze_static: true,  static_lr_scale: 0.0}
    - {name: stage3_joint_finetune, epochs: 25, lr: 1.0e-4, freeze_static: false, static_lr_scale: 0.1}
```

阶段切换时自动重建该 run 的优化器/调度器，并切换静态分支的冻结状态。

> **一个入口三种用法**：单模型单阶段 / 单模型多阶段 / 多模型共享管道。
> `train_dual.py` 是历史入口（只做单模型三阶段），保留是为了
> `run_ablation.sh` 不失效；新项目建议直接用 `train_multi.py`。

产物：

```
outputs/checkpoints/multi_arch_latest.pth      组合断点（含全部模型 + 优化器状态，可续训）
outputs/checkpoints/arch_cn_best.pth           ConvNeXt 最佳（标准格式，分析脚本可直接读）
outputs/checkpoints/arch_dcnn_best.pth         DCNN 最佳
outputs/logs/multi_arch_training_log.csv       逐轮指标，每个模型 5 列（含阶段号）
```

### batch_size 与 workers：当前取值一览

全部配置**已对齐**（本框架 / 复刻仓库 / v1.0 三者一致）：

| 位置 | batch_size | num_workers | page_workers | segment_workers |
| :--- | ---: | ---: | ---: | ---: |
| `configs/dual_stream.yaml` | 192 | 16 | 8 | 8 |
| `configs/multi_arch.yaml` | 192 | 16 | 8 | 8 |
| `configs/convnext.yaml` | 192 | 16 | 8 | 8 |
| 复刻仓库 `configs/default.yaml` | 192 | 16 | 8 | 8 |
| 代码默认值（`trainer_*.py`） | 192 | 16 | 8 | 8 |
| 评估：1-test（每轮） | 128 | min(8, num_workers) | 8 | 8 |
| 评估：20-test / 特征导出 | 256 | 8 | 8 | 8 |

**四个参数各自管什么：**

| 参数 | 何时生效 | 说明 |
| :--- | :--- | :--- |
| `batch_size` | 训练全程 | 影响显存与收敛。192 对齐 v1.0，对应 lr=5e-4。**改 batch 要按比例调 lr。** |
| `num_workers` | 每个 epoch | DataLoader 取数进程数，CPU 管道的并行度，**决定 epoch 时长**。 |
| `page_workers` | 仅数据集构建时 | `load_wptt_page` 的进程池，**只在启动时用一次**，之后销毁。 |
| `segment_workers` | 仅数据集构建时 | `pseudo_segment` 的进程池，同样只在启动时用一次。 |

> 所以：**启动慢**调 `page_workers` / `segment_workers`；**epoch 慢**调 `num_workers`。

**共享管道的一个额外好处**：两个模型共用**同一套** `num_workers`（16），
而不是各开 16 个（共 32）。因为瓶颈在 CPU，这既省核数也避免互相抢。

**显存不够时**：降 `batch_size`（两个模型共享，同步受影响，公平性不变）。
192 → 96 大致把激活显存减半，lr 按线性缩放规则减到 2.5e-4。

### DCNN 与原论文的逐项对照

| 论文要求 | `dcnn_backbone.py` |
| :--- | :--- |
| 5 卷积层，各接 2×2/stride2 最大池化 | conv1..conv5 各接 `MaxPool2d(2,2)` |
| 首层 C3，其余 C2 | k=3，然后 k=2×4 |
| 通道 80 起，每次池化 +80 | 80 / 160 / 240 / 320 / 400 |
| FC 480 → 512 | `fc1` 1600→480，`fc2` 480→512 |
| 96×96 输入，54×54 居中 | `(B,64,96,96)`，`_pad_to_96` pad 21（54+21+21=96） |
| ReLU + softmax | ReLU + 输出层 |

2.609 M 参数；权重键名（`conv1.0.*` … `output.1.*`）与
`DeepWriterID-Replication/src/models/dcnn.py` 完全一致，
**那边的 `dcnn_best.pth` 可以直接加载**。

> ⚠️ 该架构**没有任何归一化层**（原论文 2015 年，BN 刚提出）。这不是 bug，
> 但意味着训练对学习率更敏感、更依赖余弦退火兜住。复刻仓库把
> `CosineAnnealingLR` 列为「稳定收敛到 95.09% 的关键因素之一」，与此吻合。

### 从零训练：两个模型都必须关掉预训练

ConvNeXt V2-Femto 有 ImageNet FCMAE 预训练权重，**DCNN 没有任何预训练可用**。
若一边预训练、一边从零，对比里就混进了「预训练」这个变量。

```yaml
model:
  pretrained: false        # 必须 false —— 否则 ConvNeXt 会加载 ImageNet 权重
```

### 优化器配方的取舍（需要你决定）

| 做法 | 优点 | 缺点 |
| :--- | :--- | :--- |
| **各自原配方**（默认）：DCNN 用 Adam 无 wd 无 warmup；ConvNeXt 用 AdamW+wd0.05+warmup | DCNN 更容易复现到 95.09% | 架构对比里混入了优化器变量 |
| **统一配方**：两者都用 AdamW+wd0.05+warmup=1 | 严格的单变量架构对照 | DCNN 可能因缺归一化而欠拟合 |

`configs/multi_arch.yaml` 默认给的是**各自原配方**。若要严格对照，
把两个 run 的 `optimizer` / `weight_decay` / `warmup_epochs` 改成一样即可。
有条件时两组都跑，论文里更站得住。

### 千万不要重跑复刻仓库的 `preprocess.py`

它**没有固定随机种子**，README 自己说「每次运行都会产生全新的随机划分，
测试准确率在 94.5%~95.5% 浮动」。两个模型必须用同一份 `metadata.csv`
（本包 `data/features/metadata.csv` 已核实与复刻仓库 page/split 划分完全一致）。

---

### 步骤 0：上传

除代码外，还需要（`.gitignore` 通常会忽略，建议直接 `scp`/网盘）：

```
data/features/metadata.csv                 # 数据划分（必需）
data/raw/                                  # .wptt 轨迹（必需）
pretrained/convnextv2_femto.safetensors    # ConvNeXt V2-Femto 骨干网权重（必需）
outputs/checkpoints/convnext_best_9617.pth # 96.17% 静态基线权重（强烈建议）
```

> `convnext_best_9617.pth` 是**双路融合能否超过基线的关键**。
> 配置项 `model.static_init_path` 指向它；留空则静态分支随机初始化，
> 双路融合几乎不可能超过 96.17%。

**注意 `metadata.csv` 的 `file` 列是绝对路径**，换机器后需要改成本机路径：

```bash
python - <<'PY'
import pandas as pd, os
p = 'data/features/metadata.csv'
df = pd.read_csv(p, encoding='utf-8-sig')
df['file'] = df['page'].apply(lambda x: os.path.abspath(
    f"data/raw/{'WPTT2.0-Train' if x.startswith('0') else 'WPTT2.1-Train'}/{x}.wptt"))
df.to_csv(p, index=False, encoding='utf-8-sig')
PY
```

若沿用原仓库的目录结构，也可以直接把项目放在 `metadata.csv` 里记录的那个路径下。

### 步骤 1：验证特征与整套代码（5 分钟内）

```bash
# 动态特征正确性（28 项断言，只需 numpy）
python scripts/verify_dynamic.py

# 全套件冒烟测试（含静态等价性、冻结语义、权重加载、端到端 1 epoch）
python scripts/smoke_test_dual.py
```

两项都 `FAIL 0` 再往下走。云端 Linux 可以不加 `--skip_slow`，用真实数据跑端到端。

**可选：先本地演练一遍整套流程**（极小子集，把 A0/A1/C 与分析全部走通）：

```bash
python scripts/dry_run_ablation.py            # 用子进程逐步调用真实 CLI
```

它会验证真实 CLI 路径、变体护栏、以及「分析脚本在没有纯动态模型时自动
改为 A1 vs C」。演练通过后云端基本不会再出流程问题。

### 步骤 2：实验流程（A0 不重跑；A1 热启动 → C）

**一键运行（推荐）**：

```bash
bash scripts/run_ablation.sh                     # a1 -> c -> analyze
bash scripts/run_ablation.sh a1 a0ctrl c analyze # 含干净归因对照
bash scripts/run_ablation.sh a1                  # 只跑 A1
SKIP_VERIFY=1 bash scripts/run_ablation.sh       # 跳过前置验证
```

| 实验 | `--mode` | 静态流水线 | 起点 | 开销 |
| :--- | :--- | :--- | :--- | :--- |
| **A0** | —— | v2.0 原样 | —— | **不重跑**，直接引用已发表的 96.17% |
| **A1** | `static` | 修正版 | `convnext_best_9617.pth` **热启动** | 短（默认 15 轮） |
| **C** | `dual` | 修正版 | **A1 的 best** | 三阶段 5+20+25 |

三条命令等价展开：

```bash
# A1：修正流水线，从已发表的 96.17% 检查点热启动微调
python scripts/train_dual.py --mode static --prefix static_a1 \
    --static_init outputs/checkpoints/convnext_best_9617.pth \
    --static_epochs 15 --static_lr 0.4 \
    --set separator_aware=true fix_signature_layout=true

# C：双路融合，静态分支从 A1 的检查点热启动
python scripts/train_dual.py --mode dual --prefix dual_a1 \
    --static_init outputs/checkpoints/static_a1_best.pth \
    --set separator_aware=true fix_signature_layout=true
```

#### A1 为什么不需要从零训练

这是个关键的成本优化。实测：

| | A0 | A1 |
| :--- | :--- | :--- |
| 通道 0（位图） | \multicolumn{2}{c}{**逐位相同**（最大差 0.0）} |
| 通道 1..63（签名） | 退化布局 | 前缀签名布局 |

A0 与 A1 的差异**只在通道 1..63 的签名布局**上，而实测**位图是唯一具备空间
分辨力的通道**（签名通道原本退化成常数底图 + 黑白极性）。因此 A0 权重对 A1
是一次语义匹配度很高的热启动——第一层卷积在 ch0 上的权重可以直接复用。

**A1 从已收敛的检查点继续训练，收敛远快于从零训练**，默认配方是 15 轮、
`lr ≈ 2e-4`（`--static_lr 0.4` 是占位值，会被 `lr_base × batch/256` 缩放）。
从零训练才需要 50 轮 / `lr 5e-4`。

> `--static_epochs` / `--static_lr` 只影响 `static` 模式，双路的三阶段配置不受影响。

#### 关键路径

```
A1 (15 轮) ──► C (50 轮) ──► analyze
```

`A1 → C` 是**串行**的（C 需要 A1 收敛后的静态分支当冻结特征提取器），
这个依赖无法消除。但 A1 用热启动把 50 轮压到 15 轮，关键路径大幅缩短。

#### ⚠️ 方法学提醒：A1 vs 96.17% 含两个因素

A1 从 96.17% 的检查点继续训练，所以 `A1 − 96.17%` 这个差里同时包含：

1. **特征修复**带来的收益
2. **继续训练**本身的收益（即使特征不变，多训几轮也可能涨）

要干净归因，需要一组对照 **A0ctrl**：从**同一个检查点**出发、用**同样的轮数与
配方**、但保持 A0 原样特征。这样 `A1 − A0ctrl` 就只剩「特征修复」这一个变量。

```bash
bash scripts/run_ablation.sh a0ctrl        # 开销与 A1 完全相同
```

注意这一组的开销是**与 A1 相同**的，**不是**从零重跑 A0（那才是贵的那一个）。
建议加上——只多一次短跑，就能让「修复有效」这个结论站得住。

#### 论证链

| 对比 | 回答的问题 |
| :--- | :--- |
| `A1 vs 96.17%` | 修正后的静态流水线整体表现如何（含继续训练的贡献） |
| `A1 vs A0ctrl` | **纯特征修复**贡献多少（单变量，有配对 t 检验） |
| `C vs A1` | 在**修好的**静态特征上，动态分支还有增量价值吗 |
| CCA | 两个模态特征是否互补（最大典型相关 < 0.5 即高度互补） |
| 误差四象限 | 是否存在「A1 都错了、但 C 对了」的样本 |

> 还想再拆一层的话，跑 **A1-bitmap**（只开 `separator_aware`），
> 就能把 A1 的提升分成「抬笔分段修复」与「签名布局修复」两份：
> ```bash
> bash scripts/run_ablation.sh a1bitmap
> ```

产物：

```
outputs/checkpoints/static_a1_best.pth        # A1（C 的热启动来源）
outputs/checkpoints/dual_a1_best.pth          # C
outputs/checkpoints/static_a0ctrl_best.pth    # A0 对照（可选）
outputs/checkpoints/{...}_stage*.pth          # 各阶段快照
outputs/logs/{...}_training_log.csv           # 逐轮指标
```

**纯动态分支（实验 B）本方案未纳入**，但代码路径完整保留、冒烟测试覆盖。
若审稿人要求「单模态基线」，补一条即可：

```bash
python scripts/train_dual.py --mode dynamic --prefix dynamic_stream
```

> 顺带一提：动态分支只有 0.22M 参数、输入是 200×2 的序列，是这里**最便宜**的一个。

**断点续训**：默认开启。中断后重新执行同一条命令即可，会精确恢复到
「阶段 k 的第 n 轮」（含 AdamW 动量与调度器状态）。加了 `--no_resume` 才从零开始。

**训练阶段按模式分开配置**（`configs/dual_stream.yaml` 的 `training.stages_by_mode`）：

`dual` 模式：

| 阶段 | 静态分支 | 动态分支 | 融合层 | 学习率 | 轮数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 `stage1_head_warmup` | 冻结 | 训练 | 训练 | 1e-3 | 5 |
| 2 `stage2_head_converge` | 冻结 | 训练 | 训练 | 5e-4 | 20 |
| 3 `stage3_joint_finetune` | 微调 ×0.1 | 微调 | 训练 | 1e-4 | 25 |

`dynamic` 模式：同样三段（1e-3 / 5e-4 / 1e-4），但没有静态分支可冻结。

`static` 模式：**单阶段 50 轮，实际 lr = 5e-4**（`lr: 1.0` 是占位值，
会被 `lr_base × batch/256` 缩放，与原 `trainer.py` 对齐）。

> ⚠️ 这三套必须分开。若让 `static` 沿用双路的三阶段计划，它的 lr 会被
> `lr_base` 缩放成 **5e-7**，等于几乎不训练，实验 A 会得出完全错误的结论。
> 启动时打印的 `训练阶段:` 段落可以直接核对。

冻结时静态分支会被**强制切到 eval()**，避免 `drop_path` / `dropout` 在
「冻结特征提取器」上继续注入随机性——否则冻结有名无实，且每轮特征不稳定。

### 步骤 3：互补性分析

三组权重都训练完之后：

```bash
python scripts/analyze_complementarity.py \
    --static_ckpt  outputs/checkpoints/static_stream_best.pth \
    --dynamic_ckpt outputs/checkpoints/dynamic_stream_best.pth \
    --dual_ckpt    outputs/checkpoints/dual_stream_best.pth
```

输出到 `outputs/analysis/`：

| 文件 | 对应论文章节 | 说明 |
| :--- | :--- | :--- |
| `cca_correlations.csv` | 4.2 互补性分析 | 前 10 个典型相关系数；最大 < 0.5 即「高度互补」 |
| `error_quadrants.csv` / `.png` | 4.2 误差分解 | 都对 / 仅 A 对 / **仅 B 对** / 都错 |
| `significance.csv` | 4.1 主实验 | `dual_vs_static`、`static_a1_vs_a0` 两行配对 t 检验 + Cohen's d |
| `tsne_static_dynamic.png` | 4.3 可视化 | 静态 / 动态特征空间并排 t-SNE |
| `paired_20test_records.csv` | —— | 共享增强的逐轮准确率原始记录（配对检验的依据） |
| `summary.md` | —— | 汇总报告，可直接贴进论文 |

**三组实验（A0/A1/C）下，误差分解会自动改为对比 A1 vs C**——
`dual_only > 0` 的含义因此更强：**「最好的纯静态模型都错了，而融合对了」**。
这比「击败一个有缺陷的基线」硬得多。脚本会直接打印结论。

> 分析脚本会校验类别数：检查点与 metadata 必须来自同一份划分，
> 否则标签索引对不上，所有对比结论都会失效（会直接报错而不是静默出错）。
> 架构超参（`d_model` / `num_layers` / `fusion_hidden` / `pooling`）从检查点里读取，
> 所以做过超参扫描也能正确重建模型。

> ⚠️ 逐样本误差分解与 CCA 要求测试内容在多次运行间一致，脚本固定使用
> `apply_drop=False` 的确定性测试集；页面级准确率仍按页面投票计算。
> 配对 t 检验用 `run_paired_20test`：**所有模型吃同一批增强数据**，
> 配对才真正成立（分别跑两次 20-test 再配对是错的）。

### 步骤 4：核对最终成绩

三组实验的权威数字直接用 `run_ablation.sh analyze` 的输出
（`outputs/analysis/summary.md`），它已经包含：
页面级准确率、确定性测试集逐样本准确率、20-test 累积准确率与配对 t 检验。

若只想单独看某个模型的 20-test：

```bash
python -c "
import torch, pandas as pd, sys
sys.path.insert(0,'.')
from src.models.dual_stream import DualStreamWriterID
from src.evaluation.evaluator import evaluate_page_level_20test
..." 
```

更简单的方式是复用分析脚本的 `--n_test`（见上一步）。

> `scripts/evaluate_v2.py` 是 v2.0 的**纯静态**评估入口，只接受
> `ConvNeXtBackbone` 结构的检查点（键名不带 `static_branch.` 前缀），
> 不能直接评估双路模型。评估双路模型请用 `analyze_complementarity.py`。

---

## 四、实验矩阵

本次实验流程：

| 实验 | `--mode` | 静态流水线 | 起点 | 参照对象 |
| :--- | :--- | :--- | :--- | :--- |
| **A0** | —— | v2.0 原样 | —— | 已发表 96.17%，**不重跑** |
| **A0ctrl** | `static` | v2.0 原样 | 96.17% 检查点热启动 | 与 A1 完全同配方（可选） |
| **A1** | `static` | 修正版 | 96.17% 检查点热启动 | A0 / A0ctrl |
| **C** | `dual` | 修正版 | **A1 best** | **A1** |

**论证链**：`A1 vs A0ctrl` 单变量隔离出「特征修复」的贡献；
`C vs A1` 说明在**修好的**静态特征上动态分支仍有增量。
两者分开，避免把「修 bug 的收益」算进「融合的收益」。

> 原方案里的「实验 B 纯动态分支」本次不跑。**代价**是「动态特征单独有多强」
> 只能靠 CCA + 误差分解间接论证。若审稿人追问，补一条 `--mode dynamic` 即可
> （这是所有实验里最便宜的一个，0.22M 参数）。

### 可选补充消融（改配置即可，不用改代码）

```bash
# 池化方式
python scripts/train_dual.py --mode dual --prefix dual_pool_mean  --set pooling=mean
python scripts/train_dual.py --mode dual --prefix dual_pool_max   --set pooling=max

# 序列长度
python scripts/train_dual.py --mode dual --prefix dual_seq128 --set seq_len=128

# 动态分支容量
python scripts/train_dual.py --mode dual --prefix dual_d128 --set d_model=128 nhead=8

# 静态流水线贡献拆分
bash scripts/run_ablation.sh a1bitmap

# 双路也可以建在 A0 上（对比「同样的融合方法放在有缺陷的特征上」）
python scripts/train_dual.py --mode dual --prefix dual_on_a0 \
    --static_init outputs/checkpoints/static_a0ctrl_best.pth
```

---

## 五、配置项速查（`configs/dual_stream.yaml`）

| 键 | 默认 | 说明 |
| :--- | :--- | :--- |
| `mode` | `dual` | `static` / `dynamic` / `dual`，可被 `--mode` 覆盖 |
| `static_init_path` | `outputs/checkpoints/convnext_best_9617.pth` | **静态分支热启动权重** |
| `separator_aware` | `false` | `true` = 位图/签名按真实抬笔边界分段（静态流水线 v2.1） |
| `fix_signature_layout` | `false` | `true` = 按 62 维正确处理签名并逐点铺开 |
| `seq_len` | `200` | 动态序列长度 |
| `dynamic_speed_clip` | `10.0` | 归一化伪速度截断上限；`None` 关闭 |
| `dynamic_dim` | `256` | 动态特征维度 |
| `d_model` / `nhead` / `num_layers` | `64` / `4` / `4` | Transformer 配置 |
| `pooling` | `attention` | `attention` / `mean` / `max`；注意力池化仅多 65 个参数 |
| `fusion_hidden` / `fusion_dropout` | `512` / `0.3` | 融合 MLP |
| `grad_clip` | `5.0` | 梯度裁剪；阶段 1 动态分支随机初始化时防偶发大梯度 |
| `batch_size` / `num_workers` | `192` / `16` | 对齐 v1.0 |
| `page_workers` / `segment_workers` | `8` / `8` | 数据预加载进程数 |

任意键都可用 `--set key=value` 覆盖，例如
`--set batch_size=128 seq_len=300 d_model=96`。

---

## 六、模型规模

| 分支 | 参数量 |
| :--- | ---: |
| 静态 ConvNeXt V2-Femto + 512 维特征头 | 5.10 M |
| 动态 Transformer（d_model=64, 4 层） | 0.22 M |
| 其中注意力池化 | **65** |
| 融合 MLP | 0.40 M |
| **双路合计** | **5.72 M** |

动态分支只占整体 3.8%，却提供了与静态分支互补的信息——这个比例本身就是
很好的论文素材。

---

## 七、常见问题

**Q：`RuntimeError: DataLoader worker (pid xxx) is killed by signal`**
内存不足。调小 `num_workers`、`page_workers`、`segment_workers`，或调小 `batch_size`。

**Q：双路融合没有超过基线**
按顺序检查：
1. `static_init_path` 是否指向 **A1** 的检查点？启动日志里应有
   `✅ 静态流水线变体一致（A1），热启动语义匹配。`
   若出现 `⚠️ 静态流水线变体不匹配！` 说明用错了检查点（多半是拿了 A0 的），
   此时静态分支的初始化语义是错的，准确率会莫名其妙偏低。
2. `📦 静态分支权重: 已加载 144 个张量` 是否出现？缺失会打印缺失清单。
3. 阶段 1/2 的 `freeze_static` 是否为 `true`？若一开始就联合训练，
   动态分支的大梯度会冲坏预训练权重。
4. `patience=10` 是否让阶段 2 提前停了？可调大到 15。

**Q：A1 用 A0 的检查点热启动，会不会"污染"初始化？**
不会，这是**推荐做法**。护栏会打印 `ℹ️ ... 这是**推荐做法**` 而不是警告。
原因：A0 与 A1 的**第 0 通道（位图）逐位相同**，而位图是实测唯一具备空间
分辨力的通道，所以第一层卷积在 ch0 上的权重可以直接复用；纯静态模式全程可训练，
错配的 1..63 通道会被自动适配掉，而且比重头训练收敛快得多。

**反例是双路模式**：阶段 1/2 会冻结静态分支，错配的初始化在那 25 个 epoch 里
一直产出错误特征。所以 `C` 必须从 `A1` 的检查点热启动——此时护栏会打印
`⚠️ 静态流水线变体不匹配！` 并说明原因。

**Q：A1 会不会比 96.17% 高，但其实是"继续训练"的功劳？**
有可能，这正是 `A0ctrl` 对照存在的意义：同一个检查点、同样的轮数/配方，
只差特征流水线。`A1 − A0ctrl` 才是纯粹的「特征修复」贡献。
只多一次与 A1 等长的短跑，比从零重跑 A0 便宜得多。

**Q：`--mode static` 的学习率是不是太小了？**
启动时看 `训练阶段:` 段落：`static` 模式应当是**单阶段 `static_baseline`**。
实际 lr = `stages_by_mode.static[0].lr × lr_base × batch/256`。
默认 `lr: 1.0`（从零训练配方）→ 5e-4；热启动微调建议 `--static_lr 0.4` → 2e-4。
若看到 `stage1_head_warmup` 之类，说明纯静态误用了双路的三阶段计划，
lr 会被缩放成 5e-7 量级、等于几乎不训练。本仓库已按模式分开配置
（`training.stages_by_mode`），不会出现这种情况。

**Q：`fork` 相关报错**
已修好。`evaluator.py` 与 `trainer_dual.py` 里的多进程上下文现在是平台自适应的
（Linux/macOS 用 `fork`，Windows 自动退回 `spawn`），无需手工改代码。

**Q：Windows 本地跑训练报 `PermissionError`**
见下一节。

---

## 八、本地 Windows 说明

本机（开发环境）的 `outputs/`、`src/`、`scripts/` 等**既有目录**对命令行进程
拒绝新建文件（DSH 沙箱的文件权限限制）。具体表现：

- 通过编辑器/文件工具**改写已有文件**正常；
- 命令行进程在**既有目录里新建文件**被拒绝（`WinError 5`）；
- 在项目根目录下**新建目录**正常，且新目录内可自由读写。

因此：

- `scripts/verify_dynamic.py` 和 `scripts/smoke_test_dual.py --skip_slow` 可以正常跑；
- `smoke_test_dual.py` 的端到端部分默认把产物写到项目根下的 `_smoke_out/`，
  也**不需要**改动任何权限；
- 但 `scripts/train_dual.py` 默认写 `outputs/`，本地会失败。若确实想在本地跑：

```powershell
python scripts/train_dual.py --mode dual `
  --set checkpoint.save_dir=_local_out/checkpoints logging.log_dir=_local_out/logs
```

云端 Linux 不受此限制，保持默认 `outputs/` 即可。

---

## 九、复现性

- 三种模式共用同一 `seed`（默认 42），启动时统一设置
  `random` / `numpy` / `torch`（含 CUDA）随机种子。
- 每个 DataLoader worker 用 `base_seed + worker_id` 独立播种，
  保证 DropSegment 真随机且可复现。
- 动态特征提取是纯确定性函数（无随机数），冒烟测试第 5 项断言两次调用
  逐元素相同。
- 20-test 的逐轮准确率会落盘为 CSV，配对 t 检验直接读它，结果可追溯。
