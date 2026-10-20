"""
ConvNeXt V2 骨干网络：基于 timm 官方实现，适配 64 通道 96×96 输入。

论文：Woo et al., ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders, CVPR 2023.
代码：https://github.com/facebookresearch/ConvNeXt-V2 (MIT)
      https://github.com/huggingface/pytorch-image-models (Apache 2.0)

输入： [B, 64, 96, 96]
输出： [B, 512] 或 [B, num_classes]

预训练权重说明：
- 支持从本地文件加载（避免云端无法访问 HuggingFace）
- 权重来自 timm 的 convnextv2_femto.fcmae_ft_in1k
- 许可证：CC-BY-NC 4.0（仅限非商业用途）
"""

import os
import torch
import torch.nn as nn
import timm


class ConvNeXtBackbone(nn.Module):

    def __init__(
        self,
        num_classes=1019,
        in_chans=64,
        feature_dim=512,
        drop_path_rate=0.1,
        dropout_head=0.3,
        dropout_classifier=0.4,
        pretrained=True,
        pretrained_path=None,
    ):
        super().__init__()

        # 优先使用本地权重，避免云端无法访问 HuggingFace
        if pretrained and pretrained_path and os.path.exists(pretrained_path):
            print(f"📦 从本地加载预训练权重: {pretrained_path}")
            self.backbone = timm.create_model(
                'convnextv2_femto',
                pretrained=False,
                in_chans=in_chans,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )
            # 加载 safetensors 权重，跳过形状不匹配的层
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_path)
            model_dict = self.backbone.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
            skipped = [k for k in state_dict if k not in filtered]
            model_dict.update(filtered)
            self.backbone.load_state_dict(model_dict)
            print(f"✅ 已加载 {len(filtered)} 个参数，跳过 {len(skipped)} 个")
            if skipped:
                print(f"   跳过清单: {skipped}")
        else:
            # 在线下载（本地开发用，需能访问 HuggingFace）
            self.backbone = timm.create_model(
                'convnextv2_femto',
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )

        # 384 维投影到 512 维，与 v1.0 DCNN 的特征维度对齐
        self.feature_head = nn.Sequential(
            nn.Linear(384, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_head),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_classifier),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x, return_features=False):
        feat = self.backbone(x)
        features = self.feature_head(feat)

        if return_features:
            return features

        logits = self.classifier(features)
        return logits