from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset, random_split

from common.utils import sorted_alphanumeric
from classification_config import *

# 自定义数据集类
class ImageDataset(Dataset):
    # 初始化方法
    def __init__(self, main_path, label_path, transform=None):
        self.main_path = main_path
        self.transform = transform
        self.images = sorted_alphanumeric(os.listdir(main_path))
        # 读取所有的分类标签
        labels = pd.read_csv(label_path)
        # 将标签转换为数字
        self.labels_dict = dict(zip(labels['id'], labels['target']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. 读出图像文件
        img_loc = os.path.join(self.main_path, self.images[idx])
        # 2. 读出图像
        image = Image.open(img_loc).convert("RGB")
        # 3. 对图像进行变换操作
        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            # 如果没有变换操作，就抛出异常
            raise ValueError("No transform found")
        # 4. 获取标签
        img_label = self.labels_dict[idx]

        # 返回图像和标签
        return tensor_image, img_label

# 数据集加载
from torchvision import transforms
# 创建数据集
def create_dataset():
    # 数据转换器
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    # 分成训练集和测试集
    dataset = ImageDataset(main_path=IMG_PATH, label_path=FASHION_LABELS_PATH, transform=transform)
    # 划分训练集和验证集
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO])

    return train_dataset, test_dataset

train_dataset, test_dataset = create_dataset()
print(len(train_dataset))
print(len(test_dataset))