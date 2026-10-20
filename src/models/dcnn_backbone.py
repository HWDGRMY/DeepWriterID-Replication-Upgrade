"""
原论文 DCNN 骨干网（DeepWriterID, arXiv:1508.04945）。

本文件是对 `DeepWriterID-Replication/src/models/dcnn.py` 的**忠实移植**，
只做两件事：① 加上 `return_features` 以便接入双路融合框架；
② 把硬编码的 64 / 512 / 1019 变成可配置参数。
**网络结构与权重初始化方式逐行保持一致**，因此：

    · 模块名与原仓库完全相同（conv1..conv5 / fc1 / fc2 / output），
      原仓库训练出的 `dcnn_best.pth` 可以直接 load_state_dict 进来；
    · 初始化方式仍是 Kaiming Normal（conv 与 linear 都用），不做改动。

论文原文对照（Section 3）
------------------------
    "our DCNN includes 5 convolutional layers, each of which is accompanied
     by a max-pooling layer. We fix the size of the convolutional filter at
     C3 for the first layer, C2 for the others, with a stride of 1 pixel.
     The window size of max-pooling is 2x2 with a stride of 2 pixels."
    "two fully connected FC layers with 480 and 512 units in size"
    "The number of convolutional filter kernels is 80 for the first layer
     and is incremented by 80 after each max-pooling."
    "Our DCNN renders the input data into a 54x54 bitmap and puts it in the
     center of a 96x96 grid, where the extra pixels beyond the bitmap are
     the budget of padding pixels for all borders in the convolutional
     layers."

    即: M×96×96 -80C3-MP2 -160C2-MP2 -240C2-MP2 -320C2-MP2 -400C2-MP2
        -FC480 -FC512 -softmax

⚠️ 已知特性（不是 bug，是原架构如此）
------------------------------------
本网络**没有任何归一化层**（无 BatchNorm、无 LayerNorm）。2015 年 BN 刚提出，
原文大概率也没用。后果：
    · 训练对学习率更敏感，依赖 warmup / 余弦退火兜住；
      复刻仓库 README 也把 CosineAnnealingLR 列为"稳定收敛的关键因素之一"。
    · 换优化器配方（如 AdamW + weight_decay）时表现可能与原配方差异较大。
建议：复现原文时用原配方（Adam，无 weight decay，余弦退火，无 warmup）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DCNNBackbone"]


class DCNNBackbone(nn.Module):

    def __init__(
        self,
        num_classes: int = 1019,
        in_chans: int = 64,
        feature_dim: int = 512,
        dropout: float = 0.2,
        dropout_classifier: float = 0.0,
        **kwargs,                      # 吸收框架传入的无关参数（pretrained 等）
    ):
        """
        Args:
            num_classes: 类别数。
            in_chans: 输入通道数，默认 64（= 1 位图 + 63 路径签名图，
                对应论文记法里的 M）。
            feature_dim: FC2 的输出维度，论文为 512。
            dropout: FC 层之间的 dropout 强度，论文/复刻仓库为 0.2。
            dropout_classifier: 分类头前的额外 dropout。
                默认 0 以匹配原仓库（原仓库的分类头是 output = Dropout(0.2)
                + Linear，那个 0.2 已由 `dropout` 承担，不重复叠加）。
            **kwargs: 忽略（便于和 ConvNeXtBackbone 用同一套构造参数）。
        """
        super().__init__()

        self.in_chans = in_chans
        self.feature_dim = feature_dim
        self.dropout = dropout

        # ---------------- 5 层卷积，各接 2x2/stride2 最大池化 ----------------
        # 首层 C3，其余 C2；通道 80 起，每次池化后 +80
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 80, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(80, 160, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(160, 240, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(240, 320, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(320, 400, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # ---------------- 两个全连接层：480 -> 512 ----------------
        flat_dim = self._infer_flat_dim(in_chans)
        self.fc1 = nn.Sequential(
            nn.Linear(flat_dim, 480),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(480, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---------------- 分类头 ----------------
        # 命名与结构与原仓库一致：output = Dropout + Linear
        self.classifier_dropout = nn.Dropout(dropout_classifier)
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_flat_dim(in_chans: int, size: int = 96) -> int:
        """
        按论文的 96x96 输入推算展平维度（400×2×2 = 1600）。

        逐步空间尺寸（k=3/C3 首层，其余 k=2/C2，每次池化 /2）：
            96 -2 -> 94 -/2-> 47 -1 -> 46 -/2-> 23 -1 -> 22 -/2-> 11
               -1 -> 10 -/2->  5 -1 ->  4 -/2->  2
        最终 400 通道 × 2 × 2 = 1600。
        """
        h = size
        for k in (3, 2, 2, 2, 2):
            h = (h - (k - 1)) // 2
        return 400 * h * h

    def _init_weights(self):
        """与原仓库完全一致的 Kaiming Normal 初始化（conv 与 linear 都用）。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def forward(self, x, return_features: bool = False):
        """
        Args:
            x: (B, in_chans, 96, 96) —— 与 ConvNeXt 分支完全相同的输入。
            return_features: True 时返回 FC2 的 512 维特征（供双路融合使用），
                否则返回 num_classes 的 logits。

        说明: fc2 的输出维度恰好等于论文的 512，与框架的 feature_dim 默认值
        一致，因此本网络可以零改造地充当双路融合的静态分支。
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        features = self.fc2(x)          # (B, feature_dim) = (B, 512)

        if return_features:
            return features

        return self.output(self.classifier_dropout(features))
