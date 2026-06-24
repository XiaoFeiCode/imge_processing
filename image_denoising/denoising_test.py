import torch
from torch.utils.data import DataLoader

import denoising_model
import denoising_engine
import denoising_data
import denoising_config

import matplotlib.pyplot as plt



# 提取一批数据，测试
def test(model,test_loader,device):
    model.to(device)
    model.eval()

    # 获取一批数据
    data_iter = iter(test_loader)
    noisy_image, target = next(data_iter)
    print("输入图片和目标图片的形状：", noisy_image.shape, target.shape)

    # 前向传播
    with torch.no_grad():
        noisy_image = noisy_image.to(device)
        output = model(noisy_image)
        print("输出图片的形状：", output.shape)

    # 转换图像数据
    noisy_image = noisy_image.cpu().numpy().transpose(0, 2, 3, 1)
    target = target.cpu().numpy().transpose(0, 2, 3, 1)
    output = output.cpu().numpy().transpose(0, 2, 3, 1)

    # 画图显示10张图片的对比
    fig, ax = plt.subplots(3, 10, figsize=(25, 4), sharex=True, sharey=True)
    # 遍历每一行子图，对于一组图片（10张）
    for imgs, row in zip([noisy_image, output, target], ax):
        # 遍历当前行每一张图片
        for img, ax in zip(imgs, row):
            ax.imshow(img)
            ax.axis('off')
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = denoising_data.create_dataset()
    print("创建数据集完成")
    test_loader = DataLoader(test_dataset, batch_size=denoising_config.TEST_BATCH_SIZE, shuffle=False)
    print("创建数据加载器完成")
    # 创建模型,从文件加载参数
    loder_model = denoising_model.ConvDenoise()
    # 加载模型参数
    model_state_dict = torch.load(denoising_config.DENOISER_MODEL_NAME, map_location=device)
    loder_model.load_state_dict(model_state_dict)
    print("创建模型加载完成")

    # 测试
    print("测试结果如下：")
    test(loder_model,test_loader,device)

    loss = torch.nn.MSELoss()
    test_loss = denoising_engine.test_epoch(loder_model, test_loader, loss, device)
    print("测试误差为：", test_loss)




