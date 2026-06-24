import numpy as np
import torch
from matplotlib import pyplot as plt

from similarity_model import ConvEncoder, ConvDecoder
from similarity_engine import train_epoch, test_epoch, generate_embedding, find_similar_images
from similarity_data import create_dataset
from similarity_config import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 创建数据集, 从测试集中取一个数据
    dataset, train_dataset, test_dataset = create_dataset()
    test_image, _ = test_dataset[0]
    # 升维   1, 3, 68, 68 --> 3, 68, 68 --> 1, 3, 68, 68
    test_image = test_image.unsqueeze(0)

    # 2. 加载模型 先定义模型，再加载参数
    loader_encoder = ConvEncoder()
    loader_encoder.load_state_dict(torch.load(ENCODER_MODEL_NAME))
    loader_encoder.to(device)

    # 3. 生成嵌入表示
    embeddings = np.load(EMBEDDING_NAME)

    # 4. 寻找最相似的图片
    indices = find_similar_images(loader_encoder, embeddings, test_image, device)

    print(indices)

    # 5. 显示最相似的图片
    fig, axs = plt.subplots(2, 5, figsize=(25, 4))
    image_np = test_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    axs[0, 2].imshow(image_np)
    for i in range(5):
        # 取当前图片的索引号
        index = indices[0][i]
        # 从数据中取图片
        img, _ = dataset[index]
        # 转换
        img = img.permute(1, 2, 0).numpy()
        axs[1, i].imshow(img)
    # 去除坐标轴
    for ax in axs.flat:
        ax.axis('off')
    plt.show()



