# 🧭 FashionVision — 深度学习时尚图像分析平台

> 基于深度学习的时尚商品图像智能处理平台，集成**图像分类**、**图像去噪**、**以图搜图**三大核心功能，并提供直观的 Web 交互界面。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ 功能特性

| 功能 | 说明 | 技术方案 |
|------|------|----------|
| 🔍 **图像分类** | 识别时尚商品类别（上衣 / 鞋 / 包 / 裤子 / 手表） | CNN 分类器，5 分类 Softmax |
| 🧹 **图像去噪** | 对含噪图像进行智能降噪还原 | 卷积自编码器 (Denoising Autoencoder) |
| 🖼️ **以图搜图** | 上传一张图片，搜索最相似的 Top-K 商品图 | 深度卷积编码器 + KNN 余弦相似度检索 |

所有功能均通过 **Flask Web 应用** 提供可视化交互界面，拖拽上传即可使用。

---

## 📂 项目结构

```
imge_processing/
├── common/                        # 公共模块
│   ├── utils.py                   # 工具函数（随机种子、文件排序）
│   ├── fashion-labels.csv         # 24,854 条分类标签
│   └── dataset/                   # 图片数据集（需自行下载）
│       └── *.jpg
│
├── image_class/                   # 🔍 图像分类模块
│   ├── classification_model.py    # CNN 分类器定义
│   ├── classification_engine.py   # 训练/测试引擎
│   ├── classification_data.py     # 数据集加载与预处理
│   ├── classification_config.py   # 超参数配置
│   ├── classification_train.py    # 训练入口
│   ├── classification_test.py     # 测试脚本
│   └── classifier.pt              # 预训练模型权重
│
├── image_denoising/               # 🧹 图像去噪模块
│   ├── denoising_model.py         # 去噪自编码器定义
│   ├── denoising_engine.py        # 训练/测试引擎
│   ├── denoising_data.py          # 加噪数据集
│   ├── denoising_config.py        # 超参数配置
│   ├── denoising_train.py         # 训练入口
│   ├── denoising_test.py          # 测试脚本
│   └── denoiser.pt                # 预训练模型权重
│
├── image_similiar/                # 🖼️ 以图搜图模块
│   ├── similarity_model.py        # 深度卷积编码器/解码器
│   ├── similarity_engine.py       # 训练、嵌入生成、KNN 检索
│   ├── similarity_data.py         # 自监督数据集（目标=输入）
│   ├── similarity_config.py       # 超参数配置
│   ├── similarity_train.py        # 训练入口
│   ├── similarity_test.py         # 测试与可视化
│   ├── deep_encoder.pt            # 预训练编码器权重
│   └── data_embedding_f.npy       # 全量图像的嵌入向量库
│
├── web/                           # 🌐 Web 应用
│   ├── web_app.py                 # Flask 后端（端口 9000）
│   ├── templates/
│   │   └── index.html             # 前端页面（上传 + 结果展示）
│   ├── pictures/                  # 页面图标与预览图
│   └── __init__.py
│
└── test/                          # Jupyter Notebook 测试
    ├── 1_apple_test.ipynb
    ├── classification_test.ipynb
    ├── denoising_test.ipynb
    └── autoencoder.pth
```

---

## 🧠 模型架构

### 图像分类 — Classifier
```
Conv(3→8, k3) → ReLU → MaxPool(2)
 → Conv(8→16, k3) → ReLU → MaxPool(2)
 → Flatten → Linear(16×16×16 → 5)
```
- 输入：64×64 RGB 图像
- 输出：5 类 softmax 概率
- 类别：上身衣服 / 鞋 / 包 / 下身衣服 / 手表

### 图像去噪 — ConvDenoise
```
Encoder: Conv(3→32) → Conv(32→16) → Conv(16→8)   (每层 +ReLU +MaxPool)
Decoder: ConvTranspose(8→8) → ConvTranspose(8→16) → ConvTranspose(16→32) → Conv(32→3)
```
- 输入/输出：68×68 RGB 图像
- 噪声强度：σ = 0.5（高斯噪声）
- 损失函数：MSE（重建损失）

### 以图搜图 — ConvEncoder + KNN
```
Encoder: 5×[Conv → ReLU → MaxPool(2)]  (通道：3→16→32→64→128→256)
Decoder: 5×[ConvTranspose → ReLU] + Sigmoid
```
- 编码阶段输出：256×2×2 = 1024 维嵌入向量
- 检索：余弦相似度 + KNN (k=5)
- 训练策略：自监督重建（输入 = 目标），MSE 损失

---

## 📊 数据集

本项目使用 **[Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)** 数据集（Kaggle）：

- **图片数量**：约 24,854 张时尚商品图像
- **类别**：5 类（上衣、鞋、包、裤子、手表）
- **标签文件**：`common/fashion-labels.csv`

> ⚠️ **数据准备**：请将数据集图片放入 `common/dataset/` 目录下，确保图片文件名与 `fashion-labels.csv` 中的 `id` 对应。

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+（推荐 CUDA 12.1）
- 建议内存 ≥ 8GB，显存 ≥ 4GB（GPU 训练）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/XiaoFeiCode/imge_processing.git
cd imge_processing

# 2. 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 数据准备

1. 下载 [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) 数据集
2. 将下载的图片放入 `common/dataset/` 目录
3. 确保 `common/fashion-labels.csv` 已存在

### 训练模型

```bash
# 训练图像去噪模型
python -m image_denoising.denoising_train

# 训练图像分类模型
python -m image_class.classification_train

# 训练以图搜图模型（自监督编码器 + 生成嵌入向量）
python -m image_similiar.similarity_train
```

训练完成后，模型权重会自动保存到对应模块目录下。

### 启动 Web 服务

```bash
# 从项目根目录启动（推荐）
python -m web.web_app

# 或者进入 web 目录运行
cd web
python web_app.py
```

浏览器访问 **http://localhost:9000** 即可使用。

### Web 界面功能

1. **上传图片** — 点击"上传"按钮或拖拽图片到上传区
2. **图像去噪** — 点击"去噪"，对比展示加噪图与去噪结果
3. **图像分类** — 点击"分类"，显示识别出的商品类别
4. **以图搜图** — 点击"相似"，展示 Top-5 最相似的商品图片

---

## ⚙️ 配置说明

各模块的 `*_config.py` 文件包含所有可调参数：

| 参数 | 分类 | 去噪 | 图搜图 | 说明 |
|------|------|------|--------|------|
| `IMG_HEIGHT × IMG_WIDTH` | 64×64 | 68×68 | 64×64 | 输入图像尺寸 |
| `SEED` | 42 | 42 | 42 | 随机种子（可复现） |
| `LEARNING_RATE` | 1e-3 | 1e-3 | 1e-3 | AdamW 学习率 |
| `EPOCHS` | 10 | 20 | 10 | 训练轮次 |
| `TRAIN_BATCH_SIZE` | 32 | 32 | 32 | 训练批次大小 |
| `TRAIN_RATIO` | 0.8 | 0.75 | 0.8 | 训练集比例 |

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| **深度学习框架** | PyTorch, torchvision |
| **Web 后端** | Flask |
| **数值计算** | NumPy, scikit-learn |
| **图像处理** | Pillow (PIL) |
| **前端** | HTML5, CSS3, JavaScript (原生) |
| **训练辅助** | tqdm, pandas |

---

## 📋 预训练模型

项目中已包含训练好的模型权重（基于 Fashion Product Images 数据集）：

- `image_class/classifier.pt` — 分类模型
- `image_denoising/denoiser.pt` — 去噪模型
- `image_similiar/deep_encoder.pt` — 编码器（图搜图）
- `image_similiar/data_embedding_f.npy` — 全量图像嵌入向量库

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！如有问题或建议，请在 GitHub Issues 中提出。

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

<p align="center">
  <sub>Made with ❤️ and PyTorch</sub>
</p>
