"""
DataLoader 的统一构造入口 —— 训练、每轮评估、20-test 共用同一套参数。

为什么需要这个模块
------------------
早期训练和评估各自写 `DataLoader(...)`，参数很快就不一致了。实测到的三处偏差：

                batch_size   num_workers   worker_init_fn
    训练            192           16             ✓
    每轮评估        128         min(8,16)=8       ✗
    20-test         256            8              ✗

三个后果：

1. **`num_workers` 不一致 → 评估比训练慢**。训练用 16 个 worker 喂得动，
   评估只给 8 个，等于把最慢的那段限流了。而评估是纯前向、不吃梯度，
   本来就该比训练更快。

2. **`worker_init_fn` 缺失 → 20-test 的随机性失效**（这个最隐蔽）。
   `num_workers > 0` 时 PyTorch 给每个 worker 派生同一份 RNG 状态；
   20-test 靠 `apply_drop=True` 每轮重新随机丢笔，如果各 worker 的
   numpy 种子相同，不同 worker 会产生**高度相关**的 DropSegment 实现，
   20 轮里真正独立的样本数远少于 20。`worker_init_fn` 用
   `initial_seed() + worker_id` 打散，训练一直在用，评估漏了。

3. **`prefetch_factor` 没显式给**，靠默认值 2 —— 与训练碰巧一致，
   但属于「靠巧合对齐」，不算对齐。

所以这里把参数收成唯一来源，三处都调 `make_loader()`。
"""

from __future__ import annotations

import sys
import random as _random

import numpy as np
import torch
from torch.utils.data import DataLoader


#: 多进程上下文。Linux 用 fork（省内存、启动快），Windows 必须让 PyTorch
#: 自己选（fork 不可用）。
MP_CONTEXT = 'fork' if sys.platform != 'win32' else None


def loader_context(num_workers):
    """num_workers>0 时才指定多进程上下文，否则必须为 None。"""
    return MP_CONTEXT if num_workers > 0 else None


def worker_init_fn(worker_id):
    """
    每个 worker 独立随机种子。

    不设这个的话，`num_workers>0` 时所有 worker 拿到**同一份** numpy/Torch
    RNG 状态，随机增强（DropSegment / RandomRotation）会在各 worker 间
    高度相关 —— 表现为「20-test 跑了很多轮，但方差明显偏小」。
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    _random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size, num_workers, shuffle=False,
                pin_memory=True, prefetch_factor=2, drop_last=False,
                persistent_workers=False):
    """
    统一的 DataLoader 构造器。

    训练 / 评估 / 20-test 全部走这里，保证只有 batch_size、num_workers、
    shuffle 这几个「本来就该不同」的参数有差异，其余完全一致。

    Args:
        persistent_workers: 每轮训练都会重建 loader，置 True 可以免掉
            worker 反复启动的开销；但会常驻占用内存，且要求 num_workers>0。
    """
    nw = int(num_workers)
    kw = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=nw,
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        multiprocessing_context=loader_context(nw),
        worker_init_fn=worker_init_fn,
    )
    # num_workers=0 时传 prefetch_factor 会报错
    if nw > 0:
        kw['prefetch_factor'] = int(prefetch_factor)
        kw['persistent_workers'] = bool(persistent_workers)
    return DataLoader(dataset, **kw)
