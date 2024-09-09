import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# 定义初步的卷积层来处理6000个通道的输入
class InitialConv(nn.Module):
    def __init__(self, in_channels=6000, out_channels=64):
        super(InitialConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 定义全局平均池化和Softmax模块
class GlobalAvgPoolSoftmaxModule(nn.Module):
    def __init__(self, num_classes):
        super(GlobalAvgPoolSoftmaxModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # 假设输入的特征维度是512

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 定义ResNet基础模型
class ResNetBase(nn.Module):
    def __init__(self, num_classes):
        super(ResNetBase, self).__init__()
        self.initial_conv = InitialConv()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Identity()  # 移除最后的全连接层
        self.global_avg_pool_softmax = GlobalAvgPoolSoftmaxModule(num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.resnet(x)
        x = self.global_avg_pool_softmax(x)
        return x


# 损失函数模块
class DLALoss(nn.Module):
    def __init__(self, lambda1=0.5, lambda2=0.5):
        super(DLALoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, logits, targets):
        softmax_logits = F.softmax(logits, dim=1)
        log_softmax_logits = F.log_softmax(logits, dim=1)

        # 交叉熵损失
        loss1 = F.cross_entropy(logits, targets)

        # 双重损失算法
        batch_size = logits.size(0)
        max_logit = torch.max(softmax_logits, dim=1)[0]
        max_logit = max_logit.view(batch_size, 1).expand_as(logits)
        loss2 = torch.max(torch.zeros_like(logits), max_logit - logits)
        loss2 = torch.sum(loss2, dim=1).mean()

        # 最终损失
        loss = self.lambda1 * loss1 + self.lambda2 * loss2
        return loss


# 转移学习
def transfer_learning(model, num_classes):
    model.resnet.fc = nn.Identity()
    model.global_avg_pool_softmax = GlobalAvgPoolSoftmaxModule(num_classes)
    return model


if __name__ == "__main__":

    # 示例
    num_classes = 10
    model = ResNetBase(num_classes=num_classes)
    criterion = DLALoss(lambda1=0.5, lambda2=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 示例数据
    inputs = torch.randn(8, 6000, 3, 56)
    targets = torch.randint(0, num_classes, (8,))

    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, targets)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
