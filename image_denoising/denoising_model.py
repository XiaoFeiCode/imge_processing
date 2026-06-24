__all__ = ["ConvDenoise"]

# 创建模型
# 创建一个去噪自编码器
import torch
import torch.nn as nn


class ConvDenoise(nn.Module):
    # 初始化方法
    def __init__(self):
        super(ConvDenoise, self).__init__()
        # 直接一层一层的定义
        # 编码器部分
        # 定义三个卷积层
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 解码器部分 三层转置卷积层，一个普通卷积层
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    # 前向传播方法
    def forward(self, x):
        # 编码器部分
        x = self.encoder(x)
        # print("编码器输出：", x.size())
        # 解码器部分
        x = self.decoder(x)
        # print("解码器输出：", x.size())
        return x

if __name__ == '__main__':
    model = ConvDenoise()
    x = torch.randn(1, 3, 68, 68)
    print(model(x).shape)