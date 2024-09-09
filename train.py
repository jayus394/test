import torch
from torch import nn
import matplotlib.pyplot as plt
from dataloder import generate_csi_data, data_loader
from model import ResNetBase
from accuracy import Accumulator, accuracy, evaluate_accuracy_gpu
from Animator import Animator
from Timer import Timer
import argparse


# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='WiAU_heng')
parser.add_argument('-lr', type=float, default=0.1, help='learning rate for optimizer')
parser.add_argument('-batch_size', type=int, default=5, help='batch size for training')
parser.add_argument('-epoch', type=int, default=50, help='number of epochs to train')
args = parser.parse_args()

num_persons = 10
num_samples = 100
num_time = 6000
num_tx_rx = 3
num_subcarriers = 56


csi_amplitude, labels = generate_csi_data(num_persons, num_samples, num_tx_rx, num_time, num_subcarriers)

train_iter, test_iter = data_loader(num_persons, csi_amplitude, labels, batch_size)

net = ResNetBase(num_persons, num_evi)


def try_gpu(i=0):
    """如果存在则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train_ch(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.float().to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]  # 平均Loss
            train_acc = metric[1] / metric[2]  # 平均acc
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:  # 在一个epoch里取几个点
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, ' f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ' f'on {str(device)}')


if __name__ == "__main__":
    train_ch(net, train_iter, test_iter, args.epoch, args.lr, try_gpu())
    plt.show()

