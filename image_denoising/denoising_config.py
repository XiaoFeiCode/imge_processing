# 目录路径和预处理操作
IMG_PATH = '../common/dataset' # 数据集路径
IMG_H = 68 # 输入图像高度
IMG_W = 68 # 输入图像宽度

# 随机性相关配置
SEED = 42                         # 全局随机种子（确保实验可复现性）
TRAIN_RATIO = 0.75                # 训练集划分比例（75%训练，25%验证）
VAL_RATIO = 1 - TRAIN_RATIO       # 验证集比例（自动计算，无需修改）
NOISE_FACTOR = 0.5                # 设置噪声因子，用于向图像添加噪声

# 训练相关配置(超参数)
LEARNING_RATE = 1e-3              # 初始学习率（AdamW优化器使用）
EPOCHS = 20                       # 总训练轮次（需平衡过拟合与欠拟合）
TRAIN_BATCH_SIZE = 32             # 训练批次大小（GPU显存不足时可调小）
TEST_BATCH_SIZE = 32              # 验证/测试批次大小（建议与训练批次一致)

# 模型接口相关配置
PACKAGE_NAME = "image_denoising"       # 模型接口包名
DENOISER_MODEL_NAME = "denoiser.pt"    # 编码器权重保存路径（需写权限）
