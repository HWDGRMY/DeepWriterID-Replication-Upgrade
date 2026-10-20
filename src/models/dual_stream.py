"""
双路融合作者识别模型。

    静态路: ConvNeXt V2-Femto (64ch × 96×96)  → 512 维
    动态路: Transformer Encoder (200×2 伪时序) → 256 维
    融合:   768 维 → MLP → 1019 类

三种模式（用于消融实验 A/B/C）
-----------------------------
    'static'  : 纯静态，退化为原始 ConvNeXtBackbone（= 当前 v2.0 基线）
    'dynamic' : 纯动态，仅 Transformer 分支 + 线性分类头
    'dual'    : 双路拼接 + MLP 融合

    三者共用同一个类，保证数据管道、增强、评估协议完全一致，
    消融对比才具备可比性。

三阶段训练支持
-------------
    set_static_frozen(True/False) 控制静态分支是否参与梯度更新。
    冻结时静态分支同时被强制切到 eval()，避免 drop_path / dropout
    在「冻结特征提取器」上引入随机噪声。

权重兼容
--------
    `load_static_weights()` 可以直接吃旧的 ConvNeXtBackbone 检查点
    （`convnext_best_9617.pth` 等），自动补 `static_branch.` 前缀，
    因此双路模型可以站在已有 96.17% 基线的肩膀上继续训练。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from src.models.backbone import ConvNeXtBackbone
from src.models.dcnn_backbone import DCNNBackbone
from src.models.dynamic_branch import DynamicBranch
# 静态特征只有一种实现，版本号是唯一来源（见 src/data/dataset.py）
from src.data.dataset import STATIC_FEATURE_VERSION

__all__ = ["DualStreamWriterID", "build_model", "build_static_backbone",
           "STATIC_ARCHS"]

#: 支持的静态骨干网。两者接口一致（(B,C,96,96) -> 特征或 logits），
#: 所以能在同一套数据管道 / 训练器 / 评估器下互换，只换架构。
STATIC_ARCHS = ("convnext", "dcnn")


def build_static_backbone(arch, num_classes, in_chans=64, feature_dim=512,
                          drop_path_rate=0.1, dropout_head=0.3,
                          dropout_classifier=0.4, pretrained=False,
                          pretrained_path=None, dcnn_dropout=0.2,
                          dcnn_dropout_classifier=0.0):
    """
    按名字构造静态骨干网。

    两个骨干网的输入输出契约完全相同：
        输入  (B, in_chans, 96, 96)
        输出  (B, feature_dim)  —— return_features=True
              (B, num_classes)  —— 否则

    因此 DCNN 与 ConvNeXt 的对比可以排除数据管道差异，只剩架构本身。

    ⚠️ `dropout_classifier` 与 `dcnn_dropout_classifier` 必须分开
    ------------------------------------------------------------
    原论文 DCNN 的分类头**自带** `Dropout(0.2) + Linear`（在 `self.output` 里），
    没有「分类头前额外 dropout」这个概念。而 ConvNeXt 的 `classifier` 确实是
    `Sequential(Dropout(0.4), Linear)`。

    早期版本把同一个 `dropout_classifier=0.4` 也传给了 DCNN，于是它变成
    `Dropout(0.4) → Dropout(0.2) → Linear`，有效 dropout 由 0.2 涨到 0.52，
    分类头被过度正则化 —— 这是 bug，已修正为默认 0（= 与原实现完全一致）。

    参数名带 `dcnn_` 前缀，就是为了让这类「手滑复用」在代码里显形。
    """
    if arch == "convnext":
        return ConvNeXtBackbone(
            num_classes=num_classes,
            in_chans=in_chans,
            feature_dim=feature_dim,
            drop_path_rate=drop_path_rate,
            dropout_head=dropout_head,
            dropout_classifier=dropout_classifier,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        )
    if arch == "dcnn":
        if pretrained:
            print("ℹ️  DCNN 没有公开预训练权重，pretrained 参数被忽略"
                  "（从随机初始化开始）")
        return DCNNBackbone(
            num_classes=num_classes,
            in_chans=in_chans,
            feature_dim=feature_dim,
            dropout=dcnn_dropout,
            # 刻意不用 dropout_classifier：原 DCNN 没有这一层
            dropout_classifier=dcnn_dropout_classifier,
        )
    raise ValueError(f"未知的静态骨干网: {arch!r}（可选 {STATIC_ARCHS}）")


class DualStreamWriterID(nn.Module):

    def __init__(
        self,
        num_classes: int = 1019,
        mode: str = "dual",
        # ---- 静态分支 ----
        static_arch: str = "convnext",
        static_dim: int = 512,
        in_chans: int = 64,
        drop_path_rate: float = 0.1,
        dropout_head: float = 0.3,
        dropout_classifier: float = 0.4,
        pretrained: bool = False,
        pretrained_path: str | None = None,
        dcnn_dropout: float = 0.2,
        dcnn_dropout_classifier: float = 0.0,
        # ---- 动态分支 ----
        dynamic_dim: int = 256,
        seq_len: int = 200,
        dynamic_in_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dynamic_dropout: float = 0.1,
        pooling: str = "attention",
        pos_encoding: str = "sinusoidal",
        # ---- 融合层 ----
        fusion_hidden: int = 512,
        fusion_dropout: float = 0.3,
        # ---- 训练控制 ----
        freeze_static: bool = True,
    ):
        super().__init__()

        if mode not in ("static", "dynamic", "dual"):
            raise ValueError(f"不支持的模式: {mode}")

        self.mode = mode
        self.static_arch = static_arch
        self.num_classes = num_classes
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.seq_len = seq_len
        self.dynamic_in_dim = dynamic_in_dim

        # ================= 静态分支 =================
        # 注意：始终保持 self.static_branch 这一层嵌套，checkpoint 键名稳定。
        if mode in ("static", "dual"):
            self.static_branch = build_static_backbone(
                static_arch,
                num_classes=num_classes,
                in_chans=in_chans,
                feature_dim=static_dim,
                drop_path_rate=drop_path_rate,
                dropout_head=dropout_head,
                # 融合层后面已经加了 dropout，纯静态模式才需要分类头 dropout
                dropout_classifier=dropout_classifier if mode == "static" else 0.0,
                pretrained=pretrained,
                pretrained_path=pretrained_path,
                dcnn_dropout=dcnn_dropout,
                # DCNN 的分类头 dropout 单独一路，默认 0 = 与原实现一致
                dcnn_dropout_classifier=dcnn_dropout_classifier,
            )
        else:
            self.static_branch = None

        # ================= 动态分支 =================
        if mode in ("dynamic", "dual"):
            self.dynamic_branch = DynamicBranch(
                in_dim=dynamic_in_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dynamic_dropout,
                out_dim=dynamic_dim,
                seq_len=seq_len,
                pooling=pooling,
                pos_encoding=pos_encoding,
            )
        else:
            self.dynamic_branch = None

        # ================= 分类 / 融合头 =================
        if mode == "static":
            # 复用静态分支自带的 classifier，行为与 ConvNeXtBackbone 完全一致
            self.fusion = None
            self.dynamic_classifier = None
        elif mode == "dynamic":
            self.fusion = None
            self.dynamic_classifier = nn.Sequential(
                nn.Dropout(fusion_dropout),
                nn.Linear(dynamic_dim, num_classes),
            )
        else:
            fused_dim = static_dim + dynamic_dim
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, fusion_hidden),
                nn.LayerNorm(fusion_hidden),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, num_classes),
            )
            self.dynamic_classifier = None

        self._static_frozen = False
        if freeze_static and self.static_branch is not None:
            self.set_static_frozen(True)

    # ------------------------------------------------------------------
    # 冻结 / 解冻（三阶段训练用）
    # ------------------------------------------------------------------

    def set_static_frozen(self, frozen: bool):
        """冻结或解冻静态分支。冻结时同时强制 eval()，消除随机性。

        注意：纯静态模式（mode='static'）下冻结会让整个模型没有可训练参数，
        因此该请求被忽略——纯静态模型必须始终可训练。
        """
        if self.static_branch is None or self.mode == "static":
            self._static_frozen = False
            if self.static_branch is not None:
                for param in self.static_branch.parameters():
                    param.requires_grad = True
            return
        self._static_frozen = bool(frozen)
        for param in self.static_branch.parameters():
            param.requires_grad = not self._static_frozen
        if self._static_frozen:
            self.static_branch.eval()

    @property
    def static_frozen(self) -> bool:
        return self._static_frozen

    def train(self, mode: bool = True):
        """重写 train()：静态分支处于冻结状态时始终保持 eval()。

        否则 drop_path / dropout 会在被冻结的特征提取器上继续注入随机性，
        导致「冻结」名不副实、且每轮特征不稳定。
        """
        super().train(mode)
        if self._static_frozen and self.static_branch is not None:
            self.static_branch.eval()
        return self

    def trainable_parameters(self):
        """返回需要梯度的参数，供优化器使用。"""
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def _static_features(self, static_input):
        """静态路特征，冻结时关闭梯度以节省显存。"""
        with torch.set_grad_enabled(not self._static_frozen):
            return self.static_branch(static_input, return_features=True)

    def forward_features(self, static_input=None, dynamic_input=None):
        """
        返回 (static_feat, dynamic_feat)，供 CCA / t-SNE 等互补性分析使用。
        当前模式未启用的分支对应位置返回 None。
        """
        static_feat = None
        dynamic_feat = None

        if self.static_branch is not None:
            if static_input is None:
                raise ValueError("当前模式需要 static_input")
            static_feat = self._static_features(static_input)

        if self.dynamic_branch is not None:
            if dynamic_input is None:
                raise ValueError("当前模式需要 dynamic_input")
            dynamic_feat = self.dynamic_branch(dynamic_input)

        return static_feat, dynamic_feat

    def forward(self, static_input=None, dynamic_input=None,
                return_features: bool = False):
        """
        Args:
            static_input:  (B, 64, 96, 96) 静态特征图（static / dual 模式需要）
            dynamic_input: (B, seq_len, 2) 伪时序序列（dynamic / dual 模式需要）
            return_features: 为 True 时返回融合前的特征而不是 logits。
                - dual   → (B, 768) 拼接特征
                - static → (B, 512)
                - dynamic→ (B, 256)

        Returns:
            logits (B, num_classes) 或特征向量。
        """
        # 纯静态模式：完全走原路径，保证与 ConvNeXtBackbone 数值一致
        if self.mode == "static":
            if return_features:
                return self._static_features(static_input)
            return self.static_branch(static_input)

        static_feat, dynamic_feat = self.forward_features(static_input,
                                                         dynamic_input)

        if self.mode == "dynamic":
            if return_features:
                return dynamic_feat
            return self.dynamic_classifier(dynamic_feat)

        # dual
        fused = torch.cat([static_feat, dynamic_feat], dim=1)   # (B, 768)
        if return_features:
            return fused
        return self.fusion(fused)

    # ------------------------------------------------------------------
    # 权重加载
    # ------------------------------------------------------------------

    def load_static_weights(self, ckpt_path, strict: bool = False,
                            verbose: bool = True):
        """
        从旧的 ConvNeXtBackbone 检查点加载静态分支权重。

        兼容三种键名形态：
            1. {'model_state_dict': {...}} 包装（trainer.py 保存的格式）
            2. 裸 state_dict
            3. 已带 'static_branch.' 前缀（双路模型自己保存的格式）
        以及 DataParallel 的 'module.' 前缀。

        Args:
            ckpt_path: 检查点路径。
            strict: 是否要求全部键匹配。
            verbose: 是否打印加载详情。

        Returns:
            dict: {'loaded': int, 'missing': [...], 'unexpected': [...],
                   'shape_mismatch': [...]}
        """
        if self.static_branch is None:
            raise RuntimeError(f"模式 '{self.mode}' 没有静态分支，无法加载静态权重")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到检查点: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        # 统一去掉 module. 前缀，统一加上 static_branch. 前缀
        cleaned = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if not key.startswith("static_branch."):
                key = "static_branch." + key
            cleaned[key] = value

        target = self.state_dict()
        loaded, missing, unexpected, mismatch = 0, [], [], []
        for key, value in cleaned.items():
            if key not in target:
                unexpected.append(key)
                continue
            if target[key].shape != value.shape:
                mismatch.append((key, tuple(value.shape), tuple(target[key].shape)))
                continue
            target[key] = value
            loaded += 1

        for key in self.state_dict():
            if key.startswith("static_branch.") and key not in cleaned:
                missing.append(key)

        self.load_state_dict(target, strict=False)

        if strict and (missing or mismatch):
            raise RuntimeError(
                f"静态权重未完全加载: missing={missing[:5]} mismatch={mismatch[:5]}"
            )

        if verbose:
            print(f"📦 静态分支权重: 已加载 {loaded} 个张量 ← {ckpt_path}")
            if missing:
                print(f"   缺失 {len(missing)} 个（随机初始化）: {missing[:5]}")
            if mismatch:
                print(f"   形状不匹配 {len(mismatch)} 个: {mismatch[:5]}")
            if unexpected:
                print(f"   多余 {len(unexpected)} 个（已忽略）: {unexpected[:5]}")

        return {"loaded": loaded, "missing": missing,
                "unexpected": unexpected, "shape_mismatch": mismatch}


# ----------------------------------------------------------------------
# 工厂函数
# ----------------------------------------------------------------------

def read_checkpoint_feature_version(ckpt_path):
    """
    读取检查点里记录的**静态特征编码版本**，用于检测热启动的语义不匹配。

    背景：早期的 A0（v2.0 原样）与 A1（修正版）产出的 64 通道特征**语义不同**
    —— 同样叫「通道 7」，两者是完全不同的东西。用一个语义不匹配的静态权重
    去热启动，会得到一个被污染的初始化，而且**不会报错**，只是准确率莫名偏低。

    A0 分支已删除，现在只有一个特征实现，但「同一个实现内部算法改过」仍然可能
    （例如签名的尺度归一化、4×4 膨胀），所以版本号必须校验。

    Returns:
        版本字符串；检查点里没有记录时返回 None。
    """
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception:                                 # noqa: BLE001
        return None
    cfg = ckpt.get('config') if isinstance(ckpt, dict) else None
    if not isinstance(cfg, dict):
        return None
    return cfg.get('static_feature_version')


def assert_variant_compatible(ckpt_path, want_version, mode="dual", strict=False):
    """
    检查热启动检查点的特征编码版本是否与当前运行一致。

    为什么要区分 mode —— 两种情形后果完全不同：

    · **mode='static'（纯静态）**
      整个模型全程可训练，初始化只是起点，会自己适配。版本不一致时只作提示。

    · **mode='dual'（双路融合）**
      阶段 1/2 会把静态分支**冻结**，错配的初始化在这期间一直产出错误特征，
      只有阶段 3 用 0.1 倍学习率去修。**有害**，强烈警告。

    Args:
        want_version: 当前代码的特征版本（`STATIC_FEATURE_VERSION`）。
        mode: 当前训练模式。
        strict: 为 True 时不一致直接抛异常。

    Returns:
        检查点里记录的版本（可能为 None）。
    """
    got = read_checkpoint_feature_version(ckpt_path)
    is_static = (mode == 'static')

    if got is None:
        msg = (
            f"\n{'!'*74}\n"
            f"⚠️  检查点没有特征版本记录\n"
            f"    {os.path.basename(ckpt_path)} 是「特征版本号」机制出现之前保存的，"
            f"按定义属于早期语义。\n"
            f"    当前运行版本 = {want_version}，模式 = '{mode}'。\n"
            f"    早期版本的 63 个签名通道实际退化成了常数底图，"
            f"与当前实现的语义不匹配。\n"
            f"{'!'*74}"
        )
        if strict and not is_static:
            raise RuntimeError(msg)
        print(msg)
        return None

    if got != want_version:
        msg = (
            f"\n{'!'*74}\n"
            f"⚠️  特征编码版本不匹配！\n"
            f"    热启动检查点 = {got}\n"
            f"    当前运行     = {want_version}\n"
            f"    64 通道的语义可能已经改变，旧权重不可直接复用。\n"
            f"    当前模式 = '{mode}'"
            + ("（纯静态全程可训练，会自行适配，仅提示）" if is_static
               else "（双路模式下静态分支会被冻结，错配权重有害）")
            + f"\n{'!'*74}"
        )
        if strict and not is_static:
            raise RuntimeError(msg)
        print(msg)
        return got

    print(f"✅ 特征编码版本一致（{want_version}），热启动语义匹配。")
    return got


def build_model(config: dict, num_classes: int, pretrained_path: str | None = None):
    """
    依据配置字典构造模型（供训练脚本使用）。

    Args:
        config: 展平后的配置字典。
        num_classes: 类别数（由数据决定）。
        pretrained_path: 静态分支骨干网的本地预训练权重（timm safetensors）。
    """
    mode = config.get("mode", "dual")

    model = DualStreamWriterID(
        num_classes=num_classes,
        mode=mode,
        static_arch=config.get("arch", "convnext"),
        static_dim=config.get("feature_dim", 512),
        in_chans=config.get("in_chans", 64),
        drop_path_rate=config.get("drop_path", 0.1),
        dropout_head=config.get("dropout_head", 0.3),
        dropout_classifier=config.get("dropout_classifier", 0.4),
        pretrained=config.get("pretrained", False),
        pretrained_path=pretrained_path,
        dcnn_dropout=config.get("dcnn_dropout", 0.2),
        dcnn_dropout_classifier=config.get("dcnn_dropout_classifier", 0.0),
        dynamic_dim=config.get("dynamic_dim", 256),
        seq_len=config.get("seq_len", 200),
        dynamic_in_dim=config.get("dynamic_in_dim", 2),
        d_model=config.get("d_model", 64),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 4),
        dim_feedforward=config.get("dim_feedforward", 256),
        dynamic_dropout=config.get("dynamic_dropout", 0.1),
        pooling=config.get("pooling", "attention"),
        pos_encoding=config.get("pos_encoding", "sinusoidal"),
        fusion_hidden=config.get("fusion_hidden", 512),
        fusion_dropout=config.get("fusion_dropout", 0.3),
        freeze_static=config.get("freeze_static", True),
    )

    # 双路 / 纯静态模式下，从静态检查点热启动
    static_init = config.get("static_init_path")
    if static_init:
        if not os.path.isabs(static_init):
            static_init = os.path.abspath(static_init)
        if not os.path.exists(static_init):
            raise FileNotFoundError(
                f"static_init_path 指向的文件不存在: {static_init}\n"
                f"  若不想热启动静态分支，请把 static_init_path 设为 null。"
            )
        if mode in ("static", "dual"):
            assert_variant_compatible(
                static_init,
                config.get("static_feature_version", STATIC_FEATURE_VERSION),
                mode=mode,
            )
            model.load_static_weights(static_init)

    return model
