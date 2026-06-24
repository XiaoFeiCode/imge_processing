__all__ = ["train_epoch", "test_epoch"]

import torch


# 定义一个轮次epoch训练的函数
def train_epoch(model, train_loader, loss, optimizer, device):
    model.train()

    total_loss = 0.0
    for batch_idx, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        # 1. 前向传播
        output = model(input)
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
def test_epoch(model, test_loader, loss, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):
            input = input.to(device)
            target = target.to(device)
            # 1. 前向传播
            output = model(input)
            # 2. 损失计算
            loss_value = loss(output, target)
            # 3. 累计损失
            total_loss += loss_value.item()

    # 返回平均损失
    return total_loss / len(test_loader)