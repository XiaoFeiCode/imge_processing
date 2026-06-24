import torch
import denoising_model
import denoising_engine
import denoising_data
import denoising_config
# 导入进度条工具
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from common.utils import seed_everything

# 训练主流程
if __name__ == '__main__':
    # 准备操作
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(denoising_config.SEED) # 设置随机种子

    # 1. 创建数据集
    train_dataset, test_dataset = denoising_data.create_dataset()
    print("数据集大小：", len(train_dataset), len(test_dataset))
    print("=============数据集创建完成===================")

    # 2. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=denoising_config.TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=denoising_config.TEST_BATCH_SIZE, shuffle=False)
    print("============数据加载器创建完成=================")

    # 3. 创建模型, 定义损失函数，定义优化器
    model = denoising_model.ConvDenoise()
    model = model.to(device)
    loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=denoising_config.LEARNING_RATE)


    # 4. 训练模型, 保存最佳模型，看验证误差，模型保存
    # 定义验证误差 初始化为正无穷大
    min_test_loss = float('inf')
    for epoch in tqdm(range(denoising_config.EPOCHS)):
        # 调用训练轮次epoch训练的函数
        train_loss = denoising_engine.train_epoch(model, train_loader, loss, optimizer, device)
        # 调用测试轮次epoch测试的函数
        test_loss = denoising_engine.test_epoch(model, test_loader, loss, device)
        print(f'epoch: {epoch + 1}/{denoising_config.EPOCHS}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}')

        # 验证误差减小，保存模型
        if test_loss < min_test_loss:
            print("验证损失减小，保存模型...")
            min_test_loss = test_loss
            torch.save(model.state_dict(), denoising_config.DENOISER_MODEL_NAME)

    print("=============模型训练完成===================")
    print("最终验证误差为：", min_test_loss)





