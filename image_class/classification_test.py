import numpy as np
import torch
from torch.utils.data import DataLoader

import classification_model
import classification_engine
import classification_data
import classification_config

import matplotlib.pyplot as plt


# 提取一批数据，测试
def test(model, test_loader, device):
    model.to(device)
    model.eval()

    # 获取一批数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # 前向传播
    with torch.no_grad():
        images = images.to(device)
        output = model(images)
        preds = torch.argmax(output, dim=1)

    # 转换图像数据
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    labels = labels.cpu()

    # 画图显示10张图片的对比
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 画图和打印输出
    images_np = images
    labels_np = labels
    preds_np = preds
    fig, axs = plt.subplots(1, 10, figsize=(25, 4), sharey=True, sharex=True)
    for i in range(10):
        # --- 1. 获取图片 ---
        img = images_np[i]
        # 反归一化/裁剪 (防止显示噪点)
        img = np.clip(img, 0, 1)

        # --- 2. 准备文字 ---
        true_label = labels_np[i].item()
        pred_label = preds_np[i].item()

        true_name = classification_config.classification_names[true_label]
        pred_name = classification_config.classification_names[pred_label]

        # --- 3. 核心：判断对错并设置颜色 ---
        if true_label == pred_label:
            color = 'green'  # 预测正确：绿色
            title_text = f"Pred: {pred_name}\n(True: {true_name})"
        else:
            color = 'red'  # 预测错误：红色
            title_text = f"Pred: {pred_name}\n(True: {true_name})"

        # --- 4. 绘图 ---
        axs[i].imshow(img)
        axs[i].axis("off")  # 关掉坐标轴

        # 设置标题：字体大小、颜色、加粗
        axs[i].set_title(title_text, color=color, fontsize=10, fontweight='bold')

    plt.tight_layout()  # 自动调整间距，防止文字重叠
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = classification_data.create_dataset()
    print("创建数据集完成")
    test_loader = DataLoader(test_dataset, batch_size=classification_config.TEST_BATCH_SIZE, shuffle=False)
    print("创建数据加载器完成")
    # 创建模型,从文件加载参数
    loder_model = classification_model.Classifier()
    # 加载模型参数
    model_state_dict = torch.load(classification_config.CLASSIFIER_MODEL_NAME, map_location=device)
    loder_model.load_state_dict(model_state_dict)
    print("创建模型加载完成")

    # 测试
    print("测试结果如下：")
    test(loder_model, test_loader, device)

    loss = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = classification_engine.test_epoch(loder_model, test_loader, loss, device)
    print("测试误差为：", test_loss, "测试准确率为：", test_acc)
