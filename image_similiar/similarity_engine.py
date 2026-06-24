__all__ = ["train_epoch", "test_epoch", "generate_embedding"]

import torch


# 定义一个轮次epoch训练的函数
def train_epoch(encoder, decoder, train_loader, loss, optimizer, device):
    encoder.train()
    decoder.train()

    total_loss = 0.0
    for input, target in train_loader:
        input = input.to(device)
        target = target.to(device)
        # 1. 前向传播
        encoder_feature = encoder(input)
        output = decoder(encoder_feature)
        # 2. 计算损失
        loss_value = loss(output, target)
        # 3. 反向传播
        loss_value.backward()
        # 4. 更新参数
        optimizer.step()
        # 5. 清零梯度
        optimizer.zero_grad()
        # 6. 累计损失
        total_loss += loss_value.item()

    # 返回本轮次平均损失
    return total_loss / len(train_loader)

# 定义一次测试的函数
def test_epoch(encoder, decoder, test_loader, loss, device):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for input, target in test_loader:
            input = input.to(device)
            target = target.to(device)
            # 1. 前向传播
            encoder_feature = encoder(input)
            output = decoder(encoder_feature)
            # 2. 损失计算
            loss_value = loss(output, target)
            # 3. 累计损失
            total_loss += loss_value.item()
            num_batches += 1

    # 返回平均损失
    return total_loss / num_batches

# 对全量数据集生成图片嵌入式表达，返回一个ndarray矩阵
def generate_embedding(encoder, full_loader, device):
    encoder.eval()
    # 创建一个张量，用于保存图片的嵌入表示
    embedding = torch.empty(0)
    with torch.no_grad():
        for input, _ in full_loader:
            input = input.to(device)
            # 1. 前向传播
            encoder_feature = encoder(input).cpu()
            # 2. 拼接张量 将当前输出的 N=32, 256, 2, 2 的张量拼接到 encoder_feature 中
            embedding = torch.cat((embedding, encoder_feature), dim=0)

    # 返回嵌入表示
    embedding = embedding.reshape(embedding.shape[0], -1).numpy()

    return embedding


from sklearn.neighbors import NearestNeighbors
# 计算相似图片，输入一张图片的张量数据，返回最相似的图片的k个图片的索引
def find_similar_images(encoder, embedding, input_image, device, k=5):
    # 1. 将图像移动到设备上
    input_image = input_image.to(device)

    # 2. 前向传播, 获取图片的嵌入表示, 形状为 N=1, 256, 2, 2, 转为ndarray
    with torch.no_grad():
        encoder_feature = encoder(input_image).cpu().numpy()

    # 3. 转为二维结构，(N, 256*2*2)
    encoder_feature = encoder_feature.reshape(encoder_feature.shape[0], -1)

    # 4. 利用KNN算法寻找最相似的图片 metric='euclidean' 表示使用欧式距离
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')

    # 训练KNN模型, 在训练集上训练
    knn.fit(embedding)

    # 5. 寻找最相似的图片
    distances, indices = knn.kneighbors(encoder_feature)

    return indices.tolist()






