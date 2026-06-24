import torch
from torch.utils.data import DataLoader

from similarity_model import ConvEncoder, ConvDecoder
from similarity_engine import train_epoch, test_epoch, generate_embedding
import torchvision.transforms as T
from similarity_data import ImageDataset, create_dataset
# 导入配置文件参数
from similarity_config import *
# 导入numpy用于数据处理
import numpy as np
# 导入进度条工具
from tqdm import tqdm
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入优化器模块
import torch.optim as optim
# 导入自定义工具函数（如seed_everything）
from common import utils

# 主程序入口
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.seed_everything(SEED)

    # 1. 创建数据集
    dataset, train_dataset, test_dataset = create_dataset()
    print("数据集创建完成！")

    # 2. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    full_loader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    print("数据加载器创建完成！")

    # 3. 创建模型, 损失函数，定义优化器
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    encoder.to(device)
    decoder.to(device)
    loss = nn.MSELoss()
    # # 定义优化器（联合优化编码器和解码器参数）
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params, lr=LEARNING_RATE)

    # 4. 训练模型
    min_loss = float('inf')
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_epoch(encoder, decoder, train_loader, loss, optimizer, device)
        test_loss = test_epoch(encoder, decoder, test_loader, loss, device)
        print(f'epoch: {epoch + 1}/{EPOCHS}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}')

        # 验证误差减小，保存模型
        if test_loss < min_loss:
            print("验证损失减小，保存模型...")
            min_loss = test_loss
            torch.save(encoder.state_dict(), ENCODER_MODEL_NAME)

    print("模型训练完成！")
    print("最终验证误差为：", min_loss)

    # 5. 生成嵌入表示
    # 5.1 从文件加载最优模型
    model_parms = torch.load(ENCODER_MODEL_NAME)
    encoder.load_state_dict(model_parms)

    # 5.2 生成嵌入表示
    embedding = generate_embedding(encoder, full_loader, device)

    # 5.3 保存嵌入表示
    np.save(EMBEDDING_NAME, embedding)

    print("嵌入表示保存完成！")