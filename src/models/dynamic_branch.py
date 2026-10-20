"""
动态分支：Transformer Encoder + 注意力池化。

输入: (B, T, in_dim)   通常是 (B, 200, 2) 的伪时序序列
输出: (B, out_dim)     动态特征向量

为什么用 Transformer
--------------------
伪速度/曲率序列的关键信息在于**多尺度时间相关性**（书写节奏、停顿、转折的
长程依赖）。1D CNN 感受野有限，LSTM/GRU 梯度传播慢，Transformer 的自注意力
天然具备全局感受野，且对本任务 200 的长度而言计算量完全可接受。

为什么用注意力池化
------------------
平均池化对所有时间步一视同仁，会稀释转折处、停顿处等关键片段的贡献。
注意力池化让网络自行学习「哪些时间步更重要」，代价仅为 d_model+1 个参数
（d_model=64 时为 65 个）。

设计要点
--------
- Pre-LN（norm_first=True）：训练更稳定，配合 warmup 不易发散。
- batch_first=True：输入布局 (B, T, D)，与 DataLoader 拼接结果一致。
- 池化方式可选（attention / mean / max），便于做池化消融实验。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["PositionalEncoding", "DynamicBranch"]


class PositionalEncoding(nn.Module):
    """
    正弦位置编码（Vaswani et al., 2017）。

    自注意力本身对序列顺序无感，而伪时序的全部信息都在顺序里，
    因此位置编码不是可选项而是必需项。
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"正弦位置编码要求 d_model 为偶数，收到 {d_model}")

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, D)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.max_len = max_len

    def forward(self, x):
        # x: (B, T, D)
        t = x.size(1)
        if t > self.max_len:
            raise ValueError(
                f"序列长度 {t} 超过位置编码上限 {self.max_len}；"
                f"请增大 max_len 或减小 seq_len"
            )
        return self.dropout(x + self.pe[:, :t, :])


class DynamicBranch(nn.Module):
    """
    伪时序动态分支。

    Args:
        in_dim: 输入通道数，默认 2（归一化伪速度 + 曲率）。
        d_model: Transformer 内部维度。
        nhead: 多头注意力的头数，必须整除 d_model。
        num_layers: Encoder 层数。
        dim_feedforward: 前馈网络隐层维度。
        dropout: dropout 比例。
        out_dim: 输出特征维度。
        seq_len: 序列长度，用于位置编码上界。
        pooling: 池化方式，'attention' | 'mean' | 'max'。
        pos_encoding: 'sinusoidal' | 'learnable' | 'none'。
        norm_first: 是否使用 Pre-LN。
    """

    def __init__(
        self,
        in_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        out_dim: int = 256,
        seq_len: int = 200,
        pooling: str = "attention",
        pos_encoding: str = "sinusoidal",
        norm_first: bool = True,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")
        if pooling not in ("attention", "mean", "max"):
            raise ValueError(f"不支持的 pooling: {pooling}")
        if pos_encoding not in ("sinusoidal", "learnable", "none"):
            raise ValueError(f"不支持的 pos_encoding: {pos_encoding}")

        self.in_dim = in_dim
        self.d_model = d_model
        self.out_dim = out_dim
        self.pooling = pooling
        self.seq_len = seq_len

        # ---------- 输入投影 ----------
        self.input_proj = nn.Linear(in_dim, d_model)

        # ---------- 位置编码 ----------
        # 位置编码上界留出余量，方便后续把 seq_len 调大而不改结构
        max_len = max(seq_len * 2, 512)
        if pos_encoding == "sinusoidal":
            self.pos_enc = PositionalEncoding(d_model, max_len=max_len,
                                              dropout=dropout)
        elif pos_encoding == "learnable":
            self.pos_enc = nn.Sequential(
                _LearnablePositionalEncoding(d_model, max_len),
                nn.Dropout(dropout),
            )
        else:
            self.pos_enc = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---------- Transformer Encoder ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
            # 关闭 nested tensor 快速路径：它只在 eval 且无梯度时生效，
            # 会让 train/eval 走不同实现、带来微小数值差异。本任务序列很短，
            # 关掉它对速度没有影响，但能保证训练与推理结果一致可复现。
            enable_nested_tensor=False,
        )

        # ---------- 池化 ----------
        if pooling == "attention":
            # 注意力池化：仅额外增加 d_model + 1 个参数
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softmax(dim=1),
            )
        else:
            self.attn_pool = None

        # ---------- 输出投影 ----------
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, return_tokens: bool = False):
        """
        Args:
            x: (B, T, in_dim) 动态特征序列。
            return_tokens: 为 True 时额外返回池化前的 token 序列，便于可视化。

        Returns:
            未指定 return_tokens 时返回 (B, out_dim)；
            否则返回 ((B, out_dim), (B, T, d_model))。
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.size(-1) != self.in_dim:
            raise ValueError(
                f"输入最后一维应为 in_dim={self.in_dim}，实际为 {x.size(-1)}"
            )

        h = self.input_proj(x)              # (B, T, d_model)
        h = self.pos_enc(h)
        h = self.transformer(h)             # (B, T, d_model)

        if self.pooling == "attention":
            weights = self.attn_pool(h)          # (B, T, 1)，沿时间维归一化
            pooled = (h * weights).sum(dim=1)    # (B, d_model)
        elif self.pooling == "mean":
            pooled = h.mean(dim=1)
        else:  # max
            pooled = h.max(dim=1).values

        out = self.output_proj(pooled)      # (B, out_dim)
        if return_tokens:
            return out, h
        return out


class _LearnablePositionalEncoding(nn.Module):
    """可学习位置编码（消融用）。"""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)
        self.max_len = max_len

    def forward(self, x):
        t = x.size(1)
        if t > self.max_len:
            raise ValueError(f"序列长度 {t} 超过位置编码上限 {self.max_len}")
        return x + self.pe[:, :t, :]
