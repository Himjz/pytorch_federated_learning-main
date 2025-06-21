import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np


def load_data(name, root='dt', download=True, save_pre_data=True):
    """
    加载指定名称的数据集，当前仅支持 SelfDataSet。

    参数:
    name (str): 数据集名称，目前仅支持 'SelfDataSet'。
    root (str, 可选): 数据集存储的根目录，默认为 'dt'。
    download (bool, 可选): 是否下载数据集，默认为 True。
    save_pre_data (bool, 可选): 是否保存预处理后的数据，默认为 True。

    返回:
    tuple: 包含训练集、测试集和类别数量的元组。
    """
    # 这里仅支持 SelfDataSet
    data_dict = ['SelfDataSet']
    assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == 'SelfDataSet':
        # 定义数据转换
        transform = transforms.Compose([
            transforms.Resize((300, 300)),  # 调整图像大小
            transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图只有一个通道，调整归一化参数
        ])
        # 加载训练集
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'train'), transform=transform)
        # 加载测试集
        testset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'val'), transform=transform)

        # 将标签转换为张量
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    # 获取类别数量
    len_classes = len(trainset.classes)

    return trainset, testset, len_classes


def divide_data(num_client=1, num_local_class=10, dataset_name='SelfDataSet', i_seed=0,
                print_report=False, client_label_ratios=None):
    """
    根据指定的客户端数量和标签分布，将训练集数据划分给不同的客户端。

    参数:
    num_client (int, 可选): 客户端数量，默认为 1。
    num_local_class (int, 可选): 每个客户端的本地类别数量，默认为 10。若为 -1，则等于全局类别数量。
    dataset_name (str, 可选): 数据集名称，默认为 'SelfDataSet'。
    i_seed (int, 可选): 随机种子，用于确保结果可复现，默认为 0。
    print_report (bool, 可选): 是否打印客户端标签分布报告，默认为 False。
    client_label_ratios (list, tuple, np.ndarray, 可选): 每个客户端的标签比重，默认为 None。

    返回:
    tuple: 包含训练集配置字典和测试集的元组。
    """
    torch.manual_seed(i_seed)
    np.random.seed(i_seed)  # 设置numpy随机种子，确保结果可复现

    trainset, testset, len_classes = load_data(dataset_name, download=True, save_pre_data=False)

    num_classes = len_classes
    if num_local_class == -1:
        num_local_class = num_classes
    assert 0 < num_local_class <= num_classes, "number of local class should smaller than global number of class"

    # 处理客户端标签比重参数
    if client_label_ratios is not None:
        # 确保输入是可迭代对象
        if not isinstance(client_label_ratios, (list, tuple, np.ndarray)):
            raise ValueError("client_label_ratios must be a list, tuple, or numpy array")

        # 为每个客户端创建标签分布
        client_distributions = []
        for i in range(num_client):
            if i < len(client_label_ratios):
                ratio = client_label_ratios[i]
                # 检查比重是否有效
                if not (0 <= ratio <= 1):
                    raise ValueError(f"Label ratio for client {i} must be between 0 and 1")

                # 生成该客户端的标签分布
                if ratio == 0:
                    # 完全均匀分布
                    distribution = np.ones(num_classes) / num_classes
                elif ratio == 1:
                    # 完全随机分布
                    distribution = np.random.rand(num_classes)
                    distribution /= distribution.sum()
                else:
                    # 混合分布
                    uniform = np.ones(num_classes) / num_classes
                    random = np.random.rand(num_classes)
                    random /= random.sum()
                    distribution = (1 - ratio) * uniform + ratio * random

                client_distributions.append(distribution)
            else:
                # 默认为均匀分布
                client_distributions.append(np.ones(num_classes) / num_classes)
    else:
        # 默认所有客户端使用均匀分布
        client_distributions = [np.ones(num_classes) / num_classes for _ in range(num_client)]

    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': 0}
    config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes

    # 为每个类别准备数据索引
    for cls in range(num_classes):
        indexes = torch.nonzero(trainset.targets == cls).squeeze()
        num_datapoint = indexes.shape[0]
        indexes = indexes[torch.randperm(num_datapoint)]
        config_data[cls] = [0, indexes]  # 指针和打乱后的索引

    # 基于指定的分布为每个客户端分配数据
    total_samples = 0
    for i in range(num_client):
        user_id = f'f_{i:05d}'
        user_data_indexes = []
        distribution = client_distributions[i]

        # 确定该客户端的样本数量（简单平均分配）
        num_samples_per_client = len(trainset) // num_client
        if i < len(trainset) % num_client:
            num_samples_per_client += 1  # 处理余数，确保所有样本都被分配

        # 根据分布为客户端分配各类别的样本数量
        num_samples_per_class = np.round(distribution * num_samples_per_client).astype(int)

        # 调整可能因四舍五入导致的总数差异
        diff = num_samples_per_client - num_samples_per_class.sum()
        if diff != 0:
            # 找到差异最大的类别进行调整
            idx = np.argmax(distribution) if diff > 0 else np.argmin(distribution)
            num_samples_per_class[idx] += diff

        # 为客户端分配样本
        for cls in range(num_classes):
            if num_samples_per_class[cls] <= 0:
                continue

            # 获取该类别的可用数据
            available_data = config_data[cls][1][config_data[cls][0]:]
            samples_needed = min(num_samples_per_class[cls], len(available_data))

            if samples_needed > 0:
                user_data_indexes.extend(available_data[:samples_needed].tolist())
                config_data[cls][0] += samples_needed  # 更新指针

        user_data = Subset(trainset, user_data_indexes)
        trainset_config['users'].append(user_id)
        trainset_config['user_data'][user_id] = user_data
        total_samples += len(user_data)

    trainset_config['num_samples'] = total_samples

    # 打印客户端标签分布报告
    if print_report:
        print("\n===== 客户端标签分布报告 =====")
        for i, user in enumerate(trainset_config['users']):
            class_distribution = {}
            user_data = trainset_config['user_data'][user]
            for idx in range(len(user_data)):
                label = int(trainset.targets[user_data.indices[idx]].item())
                class_distribution[label] = class_distribution.get(label, 0) + 1

            print(f"\n客户端 {user}:")
            print(f"  随机标签比重: {client_label_ratios[i] if client_label_ratios is not None else 0}")
            print(f"  样本总数: {len(user_data)}")
            print("  标签分布:")
            for label in sorted(class_distribution.keys()):
                count = class_distribution[label]
                print(f"    标签 {label}: {count} 样本 ({count / len(user_data):.2%})")

        print("\n===== 总体分布 =====")
        global_distribution = {}
        for idx in range(len(trainset)):
            label = int(trainset.targets[idx].item())
            global_distribution[label] = global_distribution.get(label, 0) + 1
        print(f"  总样本数: {len(trainset)}")
        print("  标签分布:")
        for label in sorted(global_distribution.keys()):
            count = global_distribution[label]
            print(f"    标签 {label}: {count} 样本 ({count / len(trainset):.2%})")
        print("====================\n")

    return trainset_config, testset


if __name__ == "__main__":
    data_dict = ['SelfDataSet']

    for name in data_dict:
        # 示例：5个客户端，随机标签比重分别为0.0, 0.25, 0.5, 0.75, 1.0
        print(divide_data(
            num_client=5,
            num_local_class=5,
            dataset_name=name,
            i_seed=0,
            print_report=True,
            client_label_ratios=[0.0, 0.25, 0.5, 0.75, 1.0]
        ))