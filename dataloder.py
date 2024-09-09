import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


def generate_csi_data(num_persons, num_samples, num_tx_rx, num_time, num_subcarriers):
    """
    生成模拟的CSI数据。形状为 (num_person, num_samples, num_tx_rx, num_time, num_subcarriers)
    输出所有人的步态幅度数据，形状为(10, 100, 3, 6000, 56)和(10, 100)个标签。
    """
    csi_data = np.random.randn(num_persons, num_samples, num_tx_rx, num_time, num_subcarriers) + \"":
               1j * np.random.randn(num_persons, num_samples, num_tx_rx, num_time, num_subcarriers)

    csi_amplitude = torch.tensor(np.abs(csi_data))
    labels = np.arange(0, num_persons).reshape(-1, 1) * np.ones((1, 100))
    labels = torch.tensor(labels)

    return csi_amplitude, labels



def data_loader(num_persons, csi_amplitude, labels, batch_size):
    """
        输入所有人的步态幅度数据，形状为(10, 100, 3, 6000, 56)和(10, 100)个标签。
        输出训练集和测试集。
    """

    # 初始化空列表来存储训练集和测试集的数据
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # 对每个人的数据进行划分
    for i in range(num_persons):
        person_data = csi_amplitude[i]
        person_labels = labels[i]

        # 取前70个数据作为训练集，后30个数据作为测试集
        train_data.append(person_data[:70])
        train_labels.append(person_labels[:70])
        test_data.append(person_data[70:])
        test_labels.append(person_labels[70:])

    # 将列表转换为Tensor
    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_data = torch.cat(test_data, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # 创建TensorDataset
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    num_persons = 10
    num_samples = 10
    num_time = 6000
    num_tx_rx = 3
    num_subcarriers = 56

    batch_size = 5

    csi_amplitude, labels = generate_csi_data(num_persons, num_samples, num_tx_rx, num_time, num_subcarriers)

    train_iter, test_iter = data_loader(num_persons, csi_amplitude, labels, batch_size)

    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break

    # plot_csi_amplitude(csi_amplitude)
