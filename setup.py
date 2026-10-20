"""
DeepWriterID v2.0 安装配置。

基于 ConvNeXt V2 的笔迹识别系统，包含数据解析、特征提取、模型训练与评估。
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="deepwriterid-v2",
    version="2.0.0",
    author="HWDGRMY",
    author_email="zhouhao_oss@163.com",
    description="ConvNeXt V2-based online text-independent writer identification on CASIA-OLHWDB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HWDGRMY/DeepWriterID-v2",
    license="MIT",

    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",

    install_requires=[
        # 核心框架
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        # 现代化骨干网络
        "timm>=0.9.0",
        # 路径签名特征提取
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
        # 评估指标
        "scikit-learn>=1.3",
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
        "deep-learning",
        "convnext",
        "casia-olhwdb",
    ],

    include_package_data=True,
    zip_safe=False,
)