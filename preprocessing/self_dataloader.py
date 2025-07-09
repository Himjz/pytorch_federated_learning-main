import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np


class ModifiedSubset(Subset):
    """修改后的Subset类，支持自定义标签"""

    def __init__(self, dataset, indices, labels):
        super().__init__(dataset, indices)
        self.labels = labels

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return img, self.labels[idx]


def load_data(name, root='dt1', download=True, save_pre_data=True):
    """
    加载图像数据集

    参数:
        name (str): 数据集名称，目前仅支持'SelfDataSet'
        root (str): 数据存储根目录
        download (bool): 是否下载数据
        save_pre_data (bool): 是否保存预处理数据

    返回:
        trainset (Dataset): 训练集
        testset (Dataset): 测试集
        len_classes (int): 类别数量
    """
    data_dict = ['SelfDataSet']
    assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == 'SelfDataSet':
        # 定义数据转换
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图只有一个通道
        ])
        # 加载训练集和测试集（按ImageFolder格式）
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'train'), transform=transform)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'val'), transform=transform)

        # 将标签转换为张量
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    # 获取类别数量
    len_classes = len(trainset.classes)

    return trainset, testset, len_classes


def print_label_distribution(client_data, num_classes, title="标签分布报告"):
    """打印客户端标签分布报告"""
    print(f"\n===== {title} =====")
    for user, data in client_data.items():
        if isinstance(data, ModifiedSubset):
            labels = data.labels
        else:
            labels = torch.tensor([data.dataset.targets[idx].item() for idx in data.indices])

        label_counts = torch.bincount(labels.long(), minlength=num_classes)
        print(f"客户端 {user}:")
        for cls in range(num_classes):
            if label_counts[cls] > 0:
                print(f"  类别 {cls}: {label_counts[cls]} 个样本")
        print()


def calculate_label_accuracy(original_data, adjusted_data, num_classes):
    """计算标签调整后的准确率"""
    accuracies = {}
    for user in original_data.keys():
        original_subset = original_data[user]
        adjusted_subset = adjusted_data[user]

        original_labels = torch.tensor([original_subset.dataset.targets[idx].item() for idx in original_subset.indices])
        if isinstance(adjusted_subset, ModifiedSubset):
            adjusted_labels = adjusted_subset.labels
        else:
            adjusted_labels = torch.tensor(
                [adjusted_subset.dataset.targets[idx].item() for idx in adjusted_subset.indices])

        correct = (original_labels == adjusted_labels).sum().item()
        total = len(original_labels)
        accuracies[user] = correct / total if total > 0 else 0.0
    return accuracies


def print_label_accuracy(accuracies, label_adjustment, title="标签准确率报告"):
    """打印标签准确率报告"""
    print(f"\n===== {title} =====")
    for i, (user, acc) in enumerate(accuracies.items()):
        print(f"客户端 {user} (调整比例: {label_adjustment[i]:.4f}): 准确率 = {acc:.4f}")

    avg_accuracy = sum(accuracies.values()) / len(accuracies) if accuracies else 0.0
    print(f"平均准确率: {avg_accuracy:.4f}")


def divide_data(num_client=1, num_local_class=10, dataset_name='SelfDataSet', i_seed=0,
                print_report=False, label_adjustment=None, k=1):
    """
    划分数据集并分配给客户端，支持非IID数据分布和标签调整

    新增参数:
        k (int): 类别分配偏移量，控制非IID分布的偏移程度
    """
    torch.manual_seed(i_seed)
    np.random.seed(i_seed)

    trainset, testset, len_classes = load_data(dataset_name, root='dt3', download=True, save_pre_data=False)
    num_classes = len_classes

    assert 0 < num_local_class <= num_classes, "本地类别数量必须在1到总类别数之间"

    # 标签调整参数检查
    if label_adjustment is not None:
        assert len(label_adjustment) == num_client, "label_adjustment长度必须等于客户端数量"
        for ratio in label_adjustment:
            assert 0 <= ratio <= 1, "调整比例必须在0到1之间"
    else:
        label_adjustment = [0.0] * num_client

    trainset_config = {'users': [], 'user_data': {}, 'num_samples': 0}
    config_division = {}  # 记录每个类别被多少客户端分配
    config_class = {}  # 记录每个客户端分配的类别
    config_data = {}  # 记录每个类别的数据索引

    # 非IID类别分配核心逻辑：固定步长为1，基于k偏移
    # 客户端i分配的类别为 [i*k, i*k+1, ..., i*k + num_local_class - 1]（超出范围自动取模）
    for i in range(num_client):
        client_id = 'f_{0:05d}'.format(i)
        config_class[client_id] = []
        start_cls = i * k  # 起始类别 = 客户端索引 * 偏移量k
        for j in range(num_local_class):
            # 计算当前类别（步长固定为1），超出总类别数时取模
            cls = (start_cls + j) % num_classes
            if cls not in config_division:
                config_division[cls] = 1
                config_data[cls] = [0, []]  # [分配指针, 数据索引列表]
            else:
                config_division[cls] += 1
            config_class[client_id].append(cls)

    # 为每个类别分配数据索引
    for cls in config_division.keys():
        # 获取该类别的所有样本索引
        indexes = torch.nonzero(trainset.targets == cls).squeeze()
        if indexes.ndim == 0:  # 处理单样本情况
            indexes = indexes.unsqueeze(0)
        num_datapoint = indexes.shape[0]
        indexes = indexes[torch.randperm(num_datapoint)]  # 随机打乱

        # 平均分配给需要该类别的客户端
        num_partition = num_datapoint // config_division[cls]
        for i_partition in range(config_division[cls]):
            if i_partition == config_division[cls] - 1:
                # 最后一个客户端分配剩余所有样本
                config_data[cls][1].append(indexes[i_partition * num_partition:])
            else:
                config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])

    # 分配数据给客户端
    total_samples = 0
    for user in config_class.keys():
        user_data_indexes = torch.tensor([])
        for cls in config_class[user]:
            # 取出该类别分配给当前客户端的索引
            user_data_index = config_data[cls][1][config_data[cls][0]]
            user_data_indexes = torch.cat((user_data_indexes, user_data_index))
            config_data[cls][0] += 1  # 移动指针，为下一个客户端分配
        user_data_indexes = user_data_indexes.squeeze().int().tolist()

        # 初始分配（原始标签）
        user_data = Subset(trainset, user_data_indexes)
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = user_data
        total_samples += len(user_data)

    trainset_config['num_samples'] = total_samples

    # 保存原始数据用于对比
    original_client_data = {user: data for user, data in trainset_config['user_data'].items()}

    # 标签调整逻辑
    for user_idx, user in enumerate(config_class.keys()):
        adjustment_ratio = label_adjustment[user_idx]
        if adjustment_ratio <= 0:
            continue  # 不调整

        user_data = trainset_config['user_data'][user]
        indices = user_data.indices

        # 获取原始标签并生成新标签
        original_labels = torch.tensor([trainset.targets[idx].item() for idx in indices])
        new_labels = original_labels.clone()

        # 随机选择需要调整的样本
        num_to_adjust = int(len(original_labels) * adjustment_ratio)
        if num_to_adjust > 0:
            indices_to_adjust = np.random.choice(len(original_labels), num_to_adjust, replace=False)
            for idx in indices_to_adjust:
                # 确保新标签与原标签不同
                possible_classes = list(range(num_classes))
                if len(possible_classes) > 1:
                    possible_classes.remove(int(original_labels[idx]))
                new_labels[idx] = np.random.choice(possible_classes)

        # 替换为带新标签的数据集
        trainset_config['user_data'][user] = ModifiedSubset(trainset, indices, new_labels)

    # 打印报告
    if print_report:
        has_adjustment = any(ratio > 0 for ratio in label_adjustment)
        if has_adjustment:
            print_label_distribution(original_client_data, num_classes, "标签调整前分布")
            print_label_distribution(trainset_config['user_data'], num_classes, "标签调整后分布")
            accuracies = calculate_label_accuracy(original_client_data, trainset_config['user_data'], num_classes)
            print_label_accuracy(accuracies, label_adjustment)
        else:
            print_label_distribution(trainset_config['user_data'], num_classes, "标签分布（未调整）")

    return trainset_config, testset


if __name__ == "__main__":
    # 测试示例：5个客户端，每个客户端2个类别，k=1（偏移量）
    print("测试非IID分布（k=1）：")
    _, _ = divide_data(
        num_client=5,
        num_local_class=2,  # 每个客户端2个类别
        dataset_name='SelfDataSet',
        i_seed=0,
        print_report=True,
        label_adjustment=[0.0, 0.2, 0.0, 0.6, 0.0],
        k=1  # 偏移量为1
    )

    # 测试k=2的情况（类别偏移更大）
    print("\n\n测试非IID分布（k=2）：")
    _, _ = divide_data(
        num_client=5,
        num_local_class=2,
        dataset_name='SelfDataSet',
        i_seed=0,
        print_report=True,
        label_adjustment=[0.0, 0.0, 0.0, 0.0, 0.0],
        k=2  # 偏移量为2
    )