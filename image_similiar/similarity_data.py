# 定义模块的公开接口，仅暴露ImageDataset类
__all__ = ["ImageDataset", "create_dataset"]

import re
import os
import torch
import torchvision.transforms as T

from torch.utils.data import Dataset
from torch.utils.data import random_split
from PIL import Image

from similarity_config import *  # 自定义配置
from common.utils import sorted_alphanumeric

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

        # X，y
        return tensor_image, tensor_image # 输入图像和目标图像都为输入图像


# 自定义创建数据集函数
def create_dataset():
    # 定义数据变换操作
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])

    # 定义数据集对象实例
    dataset = ImageDataset(main_path=IMG_PATH, transform=transform)
    # 划分训练集和验证集
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO])

    return dataset, train_dataset, test_dataset


# 测试
if __name__ == '__main__':
    dataset, train_dataset, test_dataset = create_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
    print(len( dataset))

