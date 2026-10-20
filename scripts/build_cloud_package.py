"""
把「上云所需的一切」打包成一个 zip，云端解压即用。

打进去的内容
------------
  代码      src/ scripts/ configs/ docs/ README.md setup.py environment.yml LICENSE
  数据      data/raw/（6 个 .wptt 目录）+ data/features/（metadata 已改为相对路径）
  权重      pretrained/convnextv2_femto.safetensors（骨干网预训练）
            outputs/checkpoints/convnext_best_9617.pth（96.17% 基线，A1 的热启动来源）
            outputs/checkpoints/convnext_best_9647.pth（96.47%，可选的更好起点）
  说明      RUN_ON_CLOUD.md（解压后的三步走）

关键处理
--------
1. **metadata 里的 file 列改成相对路径**（data/raw/...）。
   原始的绝对路径指向开发机的 F:\\PycharmProjects\\DeepWriterID-v2\\...，
   换到云端会全部失效。代码侧已配合改成「相对路径按项目根目录解析」。
2. 只挑必要的检查点，跳过重复的（convnext_best.pth 95.49%、
   convnext_best_fixed_9617.pth 与 9617 精度相同）。
3. 排除 __pycache__ / .pyc / .venv / .idea / 冒烟产物。

用法
----
    python scripts/build_cloud_package.py
    python scripts/build_cloud_package.py --out dist/DeepWriterID-dual-stream.zip
    python scripts/build_cloud_package.py --no-data      # 只打代码+权重（数据自己传）
    python scripts/build_cloud_package.py --keep_staging # 保留中间目录便于检查
"""

from __future__ import annotations

import os
import sys
import shutil
import argparse
import zipfile

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.abspath('.')
#: 发布名 = 复刻升级 = Replication & Upgrade。同时作为 zip 内的顶层目录名。
DEFAULT_PKG_NAME = 'DeepWriterID-Replication-Upgrade'

# ---- 打进去的代码文件/目录 -------------------------------------------------
CODE_FILES = ['README.md', 'setup.py', 'environment.yml', 'LICENSE', '.gitignore']
CODE_DIRS = ['src', 'scripts', 'configs', 'docs']

# ---- 数据 ------------------------------------------------------------------
DATA_DIRS = ['data/raw']
DATA_FILES = ['data/features/metadata.csv', 'data/features/metadata_small.csv']

# ---- 权重 ------------------------------------------------------------------
PRETRAINED_FILES = ['pretrained/convnextv2_femto.safetensors']
CKPT_FILES = [
    'outputs/checkpoints/convnext_best_9617.pth',   # 96.17% 基线
    'outputs/checkpoints/convnext_best_9647.pth',   # 96.47%
]

# ---- 排除 ------------------------------------------------------------------
EXCLUDE_DIRS = {'__pycache__', '.venv', '.idea', '.vscode', '.git',
                '.dsh-acl-reports', '_smoke_out', '_dry_out', '_logs'}
EXCLUDE_SUFFIX = ('.pyc', '.pyo')
EXCLUDE_PREFIX = ('_',)


def _ignore(dirname, names):
    out = []
    for n in names:
        if n in EXCLUDE_DIRS or n.endswith(EXCLUDE_SUFFIX):
            out.append(n)
        elif n.startswith(EXCLUDE_PREFIX) and os.path.isdir(os.path.join(dirname, n)):
            out.append(n)
    return out


def _copy(src, dst):
    if not os.path.exists(src):
        print(f"   ⚠️ 跳过（不存在）: {src}")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return True


def _size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def write_portable_metadata(src_csv, dst_csv, verify=True):
    """
    把 metadata 的 file 列改写成相对路径（data/raw/...）。

    原始值形如 F:\\PycharmProjects\\DeepWriterID-v2\\\\data\\raw\\WPTT2.0-Train\\015-P17.wptt，
    只要能从里面认出 WPTT2.x-xxx 这一段，就能重建相对路径。
    """
    df = pd.read_csv(src_csv, encoding='utf-8-sig')
    if 'file' not in df.columns:
        raise ValueError(f"{src_csv} 缺少 file 列")

    def to_rel(raw):
        s = str(raw).replace('/', os.sep).replace('\\\\', os.sep)
        parts = [p for p in s.split(os.sep) if p]
        # 找到形如 WPTT2.0-Train 的那一段，从它开始重建
        for i, p in enumerate(parts):
            if p.startswith('WPTT') and '-' in p:
                return 'data/raw/' + '/'.join(parts[i:])
        return s            # 认不出来就原样保留

    df['file'] = df['file'].map(to_rel)

    n_abs = int(df['file'].map(os.path.isabs).sum())
    if n_abs:
        print(f"   ⚠️ {n_abs} 行仍是绝对路径（未能识别），请手工检查")

    os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
    df.to_csv(dst_csv, index=False, encoding='utf-8-sig')

    if verify:
        missing = [f for f in df['file'].unique()
                   if not os.path.exists(os.path.join(ROOT, f))]
        print(f"   {os.path.basename(dst_csv)}: {len(df)} 行，"
              f"{df['file'].nunique()} 个唯一文件，"
              f"本地缺失 {len(missing)} 个")
        if missing:
            print(f"   ⚠️ 缺失示例: {missing[:3]}")
    return df


RUN_ON_CLOUD = """# 云端运行说明（解压即用）

本包已包含**代码 + 全部数据 + 骨干网预训练权重 + 基线检查点**，解压后无需再下载任何东西。

本版本聚焦两件事：**复现原论文 DCNN** 与 **ConvNeXt V2 升级版**，
两个模型共享同一条数据管道，可以单独训练，也可以一起训练。

---

## 0. 解压

```bash
unzip @@PKG@@.zip
cd @@PKG@@
ls
# 应看到: configs data docs outputs pretrained scripts src environment.yml
```

> zip 内含单一顶层目录，直接解压即可，**不需要先 mkdir**。
> 若服务器没装 unzip：`apt-get install -y unzip`，
> 或用 `python -c "import zipfile;zipfile.ZipFile('@@PKG@@.zip').extractall('.')"`。

## 1. 安装依赖

```bash
# 方式 A：conda（推荐）
conda env create -f environment.yml
conda activate deepwriterid-v2

# 方式 B：已有环境，直接补依赖
pip install torch torchvision timm signatory safetensors \\
            numpy pandas opencv-python pyyaml tqdm matplotlib scikit-learn scipy
```

> `signatory` 在部分环境需要 `pip install signatory --no-build-isolation`。
> 装完先确认能看到 GPU：`python -c "import torch;print(torch.cuda.is_available())"`

## 2. 先验证（几分钟，别跳过）

```bash
python scripts/smoke_test_dual.py --skip_slow    # 全套件 56 项断言
python scripts/diagnose_static_features.py       # 静态流水线体检（留档用）
```

两项都 `FAIL 0` 再往下走。

## 3. 训练

```bash
# 看看配置里有哪些模型
python scripts/train_multi.py --list

# 一起训练（ConvNeXt + DCNN 共享同一份内存数据）
python scripts/train_multi.py

# 或者只单独训练其中一个
python scripts/train_multi.py --only dcnn
python scripts/train_multi.py --only convnext
```

**长跑请用 nohup 挂后台**，否则 SSH 断开就白跑了：

```bash
nohup python -u scripts/train_multi.py > train.log 2>&1 &
tail -f train.log          # Ctrl+C 只退出查看，不影响训练
```

中断后**重跑同一条命令**即可断点续训（组合断点里含优化器与调度器状态）。

### 机器资源不够时

```bash
# 显存不够：降 batch（lr 按线性缩放规则同步减半）
python scripts/train_multi.py --set batch_size=96

# CPU 核少：降 worker 数（启动慢调 page/segment，epoch 慢调 num_workers）
python scripts/train_multi.py --set num_workers=8 page_workers=4 segment_workers=4
```

## 4. 看结果

训练过程中产生的**所有指标都已经写进 CSV**：

```
outputs/logs/{group}_training_log.csv   逐轮指标：每个模型 14 列
                                        （train/test loss、char acc、char top5、
                                          page acc、page top5、best、lr、
                                          grad_norm、显存峰值…）
                                        外加 5 个全局列：耗时、compute/data 时间拆分
outputs/logs/{group}_summary.csv        每个模型一行的小结（论文做表用）
outputs/checkpoints/{group}_config.json 解析后的完整配置 + 运行环境（复现性存档）
outputs/checkpoints/*_best.pth          每个模型的最佳权重
outputs/checkpoints/{group}_latest.pth  组合断点（可续训）
```

> `compute_time_sec` vs `data_time_sec` 两列可以直接看出瓶颈在哪：
> 若 `data_time` 长期大于 `compute_time`，说明卡在 CPU 数据管道（GPU 挨饿），
> 这时**再开第二个进程并排跑不会有收益**，因为抢的是同一批 CPU 核。

## 包内容清单

| 路径 | 说明 |
| :--- | :--- |
| `src/` | 全部源码（静态流水线修正开关 + 两个骨干网 + 统一训练器） |
| `scripts/train_multi.py` | **统一训练入口**：单独训练或一起训练 |
| `configs/multi_arch.yaml` | 两个模型的配置 |
| `docs/` | 技术说明（缺陷量化证据、实验设计、FAQ） |
| `data/raw/` | 5093 个 `.wptt`，6 个目录（WPTT2.0/2.1/2.2 × Train/Test） |
| `data/features/metadata.csv` | 数据划分，`file` 列已改为**相对路径** |
| `pretrained/convnextv2_femto.safetensors` | ConvNeXt V2-Femto 骨干网权重 |
| `outputs/checkpoints/convnext_best_9617.pth` | 96.17% 纯静态基线 |
| `outputs/checkpoints/convnext_best_9647.pth` | 96.47% 纯静态基线 |

> `metadata.csv` 的 `file` 列是相对路径，代码会**相对项目根目录**解析，
> 所以从任何工作目录调用都没问题，不需要再改路径。

## 排错

见 `docs/` 下的技术说明。最常遇到的三个：

1. **`⚠️ 静态流水线变体不匹配！`** —— 热启动用错了检查点。
   A0 与 A1 的 64 通道特征语义不同，混用会让准确率莫名其妙偏低。
2. **启动时 `训练阶段:` 显示的不是你想要的阶段** —— 检查 `configs/multi_arch.yaml`
   里对应 run 的 `stages`；没写 `stages` 就是单阶段，用 run 顶层的 `lr`。
3. **`DataLoader worker killed`** —— 内存不足，调小 `num_workers` /
   `page_workers` / `segment_workers`。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', type=str, default=DEFAULT_PKG_NAME,
                    help='发布名（同时作为 zip 内的顶层目录名）')
    ap.add_argument('--out', type=str, default=None,
                    help='输出 zip 路径，默认 dist/<name>.zip')
    ap.add_argument('--no-data', action='store_true',
                    help='不打包 data/raw（数据自行上传，包会小很多）')
    ap.add_argument('--keep_staging', action='store_true',
                    help='保留中间 staging 目录')
    ap.add_argument('--staging', type=str, default='_pkg_build')
    args = ap.parse_args()

    pkg_name = args.name
    out_path = args.out or f'dist/{pkg_name}.zip'

    staging = os.path.abspath(args.staging)
    pkg_root = os.path.join(staging, pkg_name)

    if os.path.isdir(staging):
        shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(pkg_root, exist_ok=True)

    print("=" * 74)
    print(f"构建云端包: {pkg_name}")
    print(f"输出      : {out_path}")
    print(f"staging   : {pkg_root}")
    print("=" * 74)

    # ---------- 代码 ----------
    print("\n[1/5] 代码")
    for f in CODE_FILES:
        _copy(os.path.join(ROOT, f), os.path.join(pkg_root, f))
    for d in CODE_DIRS:
        _copy(os.path.join(ROOT, d), os.path.join(pkg_root, d))
        print(f"   + {d}/")

    # ---------- data/features（metadata 改成相对路径） ----------
    print("\n[2/5] data/features（file 列改为相对路径）")
    for f in DATA_FILES:
        src = os.path.join(ROOT, f)
        if not os.path.exists(src):
            print(f"   ⚠️ 跳过（不存在）: {f}")
            continue
        write_portable_metadata(src, os.path.join(pkg_root, f))

    # ---------- data/raw ----------
    print("\n[3/5] 数据")
    if args.no_data:
        print("   --no-data：跳过 data/raw（解压后需自行放置数据）")
    else:
        for d in DATA_DIRS:
            src = os.path.join(ROOT, d)
            if not os.path.exists(src):
                print(f"   ⚠️ 跳过（不存在）: {d}")
                continue
            n_files = sum(len(fs) for _r, _dd, fs in os.walk(src))
            print(f"   + {d}/  ({n_files} 个文件, {_size(src)/1e6:.1f} MB) ...")
            _copy(src, os.path.join(pkg_root, d))

    # ---------- 权重 ----------
    print("\n[4/5] 权重")
    for f in PRETRAINED_FILES + CKPT_FILES:
        src = os.path.join(ROOT, f)
        if os.path.exists(src):
            _copy(src, os.path.join(pkg_root, f))
            print(f"   + {f}  ({_size(src)/1e6:.1f} MB)")
        else:
            print(f"   ⚠️ 跳过（不存在）: {f}")

    # ---------- 说明 ----------
    print("\n[5/5] 云端说明")
    run_md = os.path.join(pkg_root, 'RUN_ON_CLOUD.md')
    with open(run_md, 'w', encoding='utf-8', newline='\n') as f:
        # 用字面量替换而不是 str.format：正文里有大量 {group} {name} 之类的
        # 占位写法，format 会把它们当成字段名而报 KeyError。
        f.write(RUN_ON_CLOUD.replace('@@PKG@@', pkg_name))
    print(f"   + RUN_ON_CLOUD.md")

    # ---------- 打包 ----------
    out_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    if os.path.exists(out_abs):
        os.remove(out_abs)

    print(f"\n压缩中 ...")
    n_files = 0
    with zipfile.ZipFile(out_abs, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for dirpath, _dirnames, filenames in os.walk(pkg_root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                arc = os.path.relpath(full, staging)
                zf.write(full, arc)
                n_files += 1

    raw_mb = _size(pkg_root) / 1e6
    zip_mb = os.path.getsize(out_abs) / 1e6
    print("\n" + "=" * 74)
    print(f"✅ 完成: {out_path}")
    print(f"   文件数   : {n_files}")
    print(f"   未压缩   : {raw_mb:.1f} MB")
    print(f"   压缩后   : {zip_mb:.1f} MB  ({zip_mb/raw_mb*100:.0f}%)")
    print("=" * 74)

    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
        print("已清理 staging 目录")
    else:
        print(f"staging 保留在: {pkg_root}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
