# 定义模块的公开接口，仅暴露ImageDataset类
__all__ = ["ImageDataset", "create_dataset"]

import re
import os
import torch
import torchvision.transforms as T

from torch.utils.data import Dataset
from torch.utils.data import random_split
from PIL import Image

from denoising_config import *  # 自定义配置

# 自定义函数，对图片文件名进行字母，数字排序
def sorted_alphanumeric(data):
    # 定义一个转换函数
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # key: 排序规则按照分离好的数字排序
    alphanum = lambda text: [convert(c) for c in re.split('([0-9]+)', text)]
    return sorted(data, key=alphanum)

# 1. 定义数据集类
class ImageDataset(Dataset):
    # 初始化方法
    def __init__(self, main_path, transform=None):
        self.main_path = main_path  # 数据集的根目录
        self.transform = transform  # 变换操作，用于数据预处理 比如改变形状
        self.images = sorted_alphanumeric(os.listdir(main_path))  # 获取数据集中的所有图像文件名，先不把全部的照片读入内存，保存为列表

    # __len__()方法返回数据集的长度，即数据集的样本数量
    def __len__(self):
        return len(self.images)

    # __getitem__()方法根据索引idx返回数据集中的一个样本
    def __getitem__(self, idx):
        # 1. 读出图像文件 拼接图片路径再加上图片名字，拼接完整路径
        img_loc = os.path.join(self.main_path, self.images[idx])

        # 2. 读出图像, 转换成RGB格式 因为默认是RGBA格式 A是透明度
        image = Image.open(img_loc).convert("RGB")

        # 3. 对图像进行变换操作,得到张量
        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            # 如果没有变换操作，就抛出异常
            raise ValueError("No transform found")
        # 4. 基于原图加入噪声
        noisy_image = tensor_image + torch.randn_like(tensor_image) * NOISE_FACTOR
        # 将噪声图像裁剪到[0,1]范围内、
        noisy_image = torch.clamp(noisy_image, 0., 1.)

        # 返回预处理后的图像张量（将噪声图片作为输入X，将原图作为目标target）
        return noisy_image, tensor_image


# 自定义创建数据集函数
def create_dataset():
    # 定义数据变换操作
    transform = T.Compose([
        T.Resize((68, 68)),
        T.ToTensor()
    ])

    # 定义数据集对象实例
    dataset = ImageDataset(main_path=IMG_PATH, transform=transform)
    # 划分训练集和验证集
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO])

    return train_dataset, test_dataset


# 测试
if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    print(len(train_dataset))
    print(len(test_dataset))

