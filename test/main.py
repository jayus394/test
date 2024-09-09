import torch
from torchsummary import summary
import torch.nn as nn


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                # 输入shape (1,28,28)
                out_channels=16,
                # 输出shape(16,28,28)，16也是卷积核的数量
                kernel_size=5,
                stride=1,
                padding=2),
            # 如果想要conv2d出来的图片长宽没有变化，那么当stride=1的时候，padding=(kernel_size-1)/2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # 在2*2空间里面下采样，输出shape(16,14,14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                # 输入shape (16,14,14)
                out_channels=32,
                # 输出shape(32,14,14)
                kernel_size=5,
                stride=1,
                padding=2),
            # 输出shape(32,7,7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Linear(32 * 7 * 7, 10)

    # 输出一个十维的东西，表示我每个数字可能性的权重

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


#model = model

model = model().to(device="cuda")
summary(model, (1, 28, 28))
