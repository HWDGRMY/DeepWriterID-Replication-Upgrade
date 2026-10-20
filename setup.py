"""
DeepWriterID Replication & Upgrade —— 安装配置。

论文《DeepWriterID》的对齐复刻，并在 CASIA-OLHWDB 2.0–2.2（1019 位书写者）上
提供 DCNN 与 ConvNeXt V2-Femto 双骨干的配对对照实验。

包含数据解析、伪字符切分、路径签名特征提取、DropSegment 增强、模型训练与
页面级评估的完整实现。

安装（开发模式）：
    pip install -e .

注意：本项目内部一律以 `from src.xxx import ...` 引用模块，因此 `src` 本身
即为包名；dist 名称（deepwriterid）仅用于 pip 元数据，不影响导入路径。
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="deepwriterid",
    version="2.2.1",
    author="HWDGRMY",
    author_email="zhouhao_oss@163.com",
    description=(
        "论文对齐的 DeepWriterID 复刻：CASIA-OLHWDB 上的双骨干"
        "（DCNN / ConvNeXt V2-Femto）在线文本无关书写者识别"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HWDGRMY/DeepWriterID-Replication-Upgrade",
    license="MIT",

    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",

    install_requires=[
        # 核心框架
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        # 现代化骨干网络
        "timm>=0.9.0",
        # 本地加载 timm 预训练权重（src/models/backbone.py 用 safetensors.torch.load_file）
        "safetensors>=0.4.0",
        # 路径签名特征提取（建议配合 --no-build-isolation 安装，见 README §13.1）
        "signatory",
        # 数据处理
        "numpy>=1.24",
        "pandas>=2.0",
        "opencv-python>=4.8",
        # 配置与日志
        "pyyaml>=6.0",
        "tqdm>=4.65",
        # 可视化
        "matplotlib>=3.7",
        # 评估指标（scikit-learn 用于 t-SNE 与混淆矩阵）
        "scikit-learn>=1.3",
        # 互补性分析的配对 t 检验（scripts/analyze_complementarity.py 用 scipy.stats）
        "scipy>=1.10",
    ],

    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],

    keywords=[
        "writer-identification",
        "handwriting-recognition",
        "online-handwriting",
        "path-signature",
        "drop-segment",
        "deep-learning",
        "convnext",
        "dcnn",
        "casia-olhwdb",
        "deepwriterid",
    ],

    include_package_data=True,
    zip_safe=False,
)
