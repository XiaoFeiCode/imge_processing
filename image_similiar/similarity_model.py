__all__ = ['ConvEncoder', 'ConvDecoder']

import torch
import torch.nn as nn
# 1. 创建编码器
class ConvEncoder(nn.Module):
    # 初始化方法
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # 五层卷积层
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.encoder(x)

class ConvDecoder(nn.Module):
    # 创建解码器
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.decoder = nn.Sequential(
            # 五层卷积层
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0, output_padding=0),
            # 将输出张量裁剪到[0,1]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

if __name__ == '__main__':
    # 创建编码器实例
    encoder = ConvEncoder()
    # 创建解码器实例
    decoder = ConvDecoder()
    # 创建输入张量
    input_tensor = torch.randn(1, 3, 64, 64)
    # 编码
    encoded_tensor = encoder(input_tensor)
    # 解码
    decoded_tensor = decoder(encoded_tensor)
    # 输出结果
    print(decoded_tensor.shape)