import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np


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
    # 这里仅支持 SelfDataSet
    data_dict = ['SelfDataSet']
    assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == 'SelfDataSet':
        # 定义数据转换
        transform = transforms.Compose([
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


def print_label_distribution(client_data, num_classes, title="标签分布报告"):
    """
    打印客户端标签分布报告

    参数:
        client_data (dict): 客户端数据字典，格式为{客户端ID: 数据集}
        num_classes (int): 总类别数量
        title (str): 报告标题
    """
    print(f"\n===== {title} =====")
    for user, data in client_data.items():
        # 获取该客户端所有样本的标签
        if isinstance(data, ModifiedSubset):  # 如果是修改后的数据集
            labels = data.labels
        else:  # 普通Subset数据集
            labels = torch.tensor([data.dataset.targets[idx].item() for idx in data.indices])

        # 统计每个类别的样本数量
        label_counts = torch.bincount(labels.long(), minlength=num_classes)

        # 打印标签分布
        print(f"客户端 {user}:")
        for cls in range(num_classes):
            if label_counts[cls] > 0:
                print(f"  类别 {cls}: {label_counts[cls]} 个样本")
        print()


def calculate_label_accuracy(original_data, adjusted_data, num_classes):
    """
    计算标签调整后的准确率

    参数:
        original_data (dict): 原始数据字典，格式为{客户端ID: 数据集}
        adjusted_data (dict): 调整后的数据字典，格式为{客户端ID: 数据集}
        num_classes (int): 总类别数量

    返回:
        dict: 每个客户端的标签准确率
    """
    accuracies = {}

    for user in original_data.keys():
        original_subset = original_data[user]
        adjusted_subset = adjusted_data[user]

        # 获取原始标签
        original_labels = torch.tensor([original_subset.dataset.targets[idx].item() for idx in original_subset.indices])

        # 获取调整后的标签
        if isinstance(adjusted_subset, ModifiedSubset):
            adjusted_labels = adjusted_subset.labels
        else:
            adjusted_labels = torch.tensor(
                [adjusted_subset.dataset.targets[idx].item() for idx in adjusted_subset.indices])

        # 计算准确率
        correct = (original_labels == adjusted_labels).sum().item()
        total = len(original_labels)
        accuracy = correct / total if total > 0 else 0.0

        accuracies[user] = accuracy

    return accuracies


def print_label_accuracy(accuracies, label_adjustment, title="标签准确率报告"):
    """
    打印标签准确率报告

    参数:
        accuracies (dict): 每个客户端的标签准确率
        label_adjustment (list): 标签调整比例列表
        title (str): 报告标题
    """
    print(f"\n===== {title} =====")
    for i, (user, acc) in enumerate(accuracies.items()):
        print(f"客户端 {user} (调整比例: {label_adjustment[i]:.4f}): 准确率 = {acc:.4f}")

    # 计算平均准确率
    avg_accuracy = sum(accuracies.values()) / len(accuracies) if accuracies else 0.0
    print(f"平均准确率: {avg_accuracy:.4f}")


def divide_data(num_client=1, num_local_class=10, dataset_name='SelfDataSet', i_seed=0,
                print_report=False, label_adjustment=None):
    """
    划分数据集并分配给客户端，支持非IID数据分布和标签调整

    参数:
        num_client (int): 客户端数量
        num_local_class (int): 每个客户端的本地类别数量，-1表示使用全部类别
        dataset_name (str): 数据集名称
        i_seed (int): 随机种子
        print_report (bool): 是否打印客户端标签分布报告
        label_adjustment (list/tuple): 标签调整策略，数组或元组，每个元素对应一个客户端的调整比例

    返回:
        trainset_config (dict): 训练集配置，包含客户端数据划分
        testset (Dataset): 测试集
    """
    torch.manual_seed(i_seed)
    np.random.seed(i_seed)  # 确保numpy随机数生成器也使用相同的种子

    trainset, testset, len_classes = load_data(dataset_name, download=True, save_pre_data=False)

    num_classes = len_classes
    if num_local_class == -1:
        num_local_class = num_classes
    assert 0 < num_local_class <= num_classes, "number of local class should smaller than global number of class"

    # 检查标签调整参数
    if label_adjustment is not None:
        assert len(label_adjustment) == num_client, "label_adjustment must have length equal to num_client"
        for ratio in label_adjustment:
            assert 0 <= ratio <= 1, "each element in label_adjustment must be between 0 and 1"

    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': 0}
    config_division = {}  # Count of the classes for division
    config_class = {}  # Configuration of class distribution in clients
    config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes

    for i in range(num_client):
        config_class['f_{0:05d}'.format(i)] = []
        for j in range(num_local_class):
            cls = (i + j) % num_classes
            if cls not in config_division:
                config_division[cls] = 1
                config_data[cls] = [0, []]
            else:
                config_division[cls] += 1
            config_class['f_{0:05d}'.format(i)].append(cls)

    # 为每个类别分配数据索引
    for cls in config_division.keys():
        indexes = torch.nonzero(trainset.targets == cls)
        num_datapoint = indexes.shape[0]
        indexes = indexes[torch.randperm(num_datapoint)]
        num_partition = num_datapoint // config_division[cls]
        for i_partition in range(config_division[cls]):
            if i_partition == config_division[cls] - 1:
                config_data[cls][1].append(indexes[i_partition * num_partition:])
            else:
                config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])

    total_samples = 0
    for user_idx, user in enumerate(config_class.keys()):
        user_data_indexes = torch.tensor([])
        for cls in config_class[user]:
            user_data_index = config_data[cls][1][config_data[cls][0]]
            user_data_indexes = torch.cat((user_data_indexes, user_data_index))
            config_data[cls][0] += 1
        user_data_indexes = user_data_indexes.squeeze().int().tolist()

        # 创建用户数据集
        user_data = Subset(trainset, user_data_indexes)
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = user_data
        total_samples += len(user_data)

    trainset_config['num_samples'] = total_samples

    # 保存调整前的数据集副本用于对比
    original_client_data = {user: data for user, data in trainset_config['user_data'].items()}

    # 标签调整
    if label_adjustment is not None:
        for user_idx, user in enumerate(config_class.keys()):
            user_data = trainset_config['user_data'][user]
            adjustment_ratio = label_adjustment[user_idx]

            if adjustment_ratio > 0:  # 需要调整标签
                # 获取当前用户数据的标签和索引
                indices = user_data.indices

                if isinstance(user_data, ModifiedSubset):  # 如果已经调整过
                    original_labels = user_data.labels
                else:  # 普通Subset数据集
                    original_labels = torch.tensor([trainset.targets[idx].item() for idx in indices])

                new_labels = original_labels.clone()

                # 计算需要调整的标签数量
                num_to_adjust = int(len(original_labels) * adjustment_ratio)
                if num_to_adjust > 0:
                    # 选择要调整的标签索引
                    indices_to_adjust = np.random.choice(len(original_labels), num_to_adjust, replace=False)

                    # 为每个选中的标签随机分配新类别
                    for idx in indices_to_adjust:
                        # 确保新标签与原标签不同（如果可能）
                        possible_classes = list(range(num_classes))
                        if len(possible_classes) > 1 and int(original_labels[idx]) in possible_classes:
                            possible_classes.remove(int(original_labels[idx]))
                        new_label = np.random.choice(possible_classes)
                        new_labels[idx] = new_label

                # 创建一个修改后的数据集副本
                trainset_config['user_data'][user] = ModifiedSubset(trainset, indices, new_labels)

    # 打印标签分布报告和准确率
    if print_report:
        if label_adjustment is not None:
            print_label_distribution(original_client_data, num_classes, "标签调整前分布")
            print_label_distribution(trainset_config['user_data'], num_classes, "标签调整后分布")

            # 计算并打印准确率
            accuracies = calculate_label_accuracy(original_client_data, trainset_config['user_data'], num_classes)
            print_label_accuracy(accuracies, label_adjustment)
        else:
            print_label_distribution(trainset_config['user_data'], num_classes, "标签分布")

    return trainset_config, testset


class ModifiedSubset(Subset):
    """修改后的Subset类，支持自定义标签"""

    def __init__(self, dataset, indices, labels):
        super().__init__(dataset, indices)
        self.labels = labels

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return img, self.labels[idx]


if __name__ == "__main__":
    data_dict = ['SelfDataSet']

    for name in data_dict:
        # 示例：划分5个客户端，每个客户端2个类别，打印分布报告，使用不同的标签调整策略
        print(f"\n测试数据集: {name}")
        _, _ = divide_data(
            num_client=5,
            num_local_class=2,
            dataset_name=name,
            i_seed=0,
            print_report=True,
            label_adjustment=[0.0, 0.3, 0.5, 0.7, 1.0]  # 不同客户端使用不同的标签调整比例
        )