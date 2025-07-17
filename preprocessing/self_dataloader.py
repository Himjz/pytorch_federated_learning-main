import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LabelControlledDataset(Dataset):
    """自定义数据集类，支持直接控制标签"""

    def __init__(self, base_dataset, indices, custom_labels=None):
        self.base_dataset = base_dataset  # 原始数据集（用于获取图像）
        self.indices = indices  # 样本索引列表
        # 初始化标签：若未提供自定义标签则使用原始标签
        if custom_labels is None:
            self.labels = [base_dataset.original_targets[idx].item() for idx in indices]
        else:
            self.labels = custom_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """返回图像和自定义标签"""
        img, _ = self.base_dataset[self.indices[idx]]  # 忽略原始标签
        return img, self.labels[idx]  # 直接返回自定义标签


def load_data(name, root='dt1'):
    """加载数据集并预处理"""
    supported_datasets = ['SelfDataSet']
    assert name in supported_datasets, f"不支持的数据集，仅支持{supported_datasets}"

    # 创建数据目录（若不存在）
    os.makedirs(root, exist_ok=True)

    if name == 'SelfDataSet':
        # 定义图像转换
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转为灰度图
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图归一化
        ])

        # 加载基础数据集（ImageFolder格式）
        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(root, 'train'),
            transform=transform
        )
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(root, 'val'),
            transform=transform
        )

        # 存储原始标签（用于后续对比和修改）
        trainset.original_targets = torch.tensor(trainset.targets, dtype=torch.long)
        testset.original_targets = torch.tensor(testset.targets, dtype=torch.long)

    # 类别数量
    num_classes = len(trainset.classes)
    return trainset, testset, num_classes


def print_label_distribution(client_data, num_classes, title="标签分布报告"):
    """打印客户端标签分布"""
    print(f"\n===== {title} =====")
    for user, data in client_data.items():
        # 统计每个类别的样本数
        labels = torch.tensor(data.labels, dtype=torch.long)
        label_counts = torch.bincount(labels, minlength=num_classes)
        print(f"客户端 {user}:")
        for cls in range(num_classes):
            if label_counts[cls] > 0:
                print(f"  类别 {cls}: {label_counts[cls]} 个样本")
        print()


def calculate_label_accuracy(original_data, adjusted_data, num_classes):
    """计算标签调整后的准确率（与原始标签对比）"""
    accuracies = {}
    for user in original_data.keys():
        original_labels = torch.tensor(original_data[user].labels, dtype=torch.long)
        adjusted_labels = torch.tensor(adjusted_data[user].labels, dtype=torch.long)

        correct = (original_labels == adjusted_labels).sum().item()
        total = len(original_labels)
        accuracies[user] = correct / total if total > 0 else 0.0
    return accuracies


def print_label_accuracy(accuracies, label_adjustment, title="标签准确率报告"):
    """打印标签准确率"""
    print(f"\n===== {title} =====")
    for i, (user, acc) in enumerate(accuracies.items()):
        print(f"客户端 {user} (目标正确率: {label_adjustment[i]:.4f}): 准确率 = {acc:.4f}")
    avg_accuracy = sum(accuracies.values()) / len(accuracies) if accuracies else 0.0
    print(f"平均准确率: {avg_accuracy:.4f}")


def divide_data(
        num_client=1,
        num_local_class=10,
        dataset_name='SelfDataSet',
        i_seed=0,
        print_report=False,
        label_adjustment=None,
        k=1
):
    """
    划分数据集并分配给客户端，支持非IID分布和标签错误标注

    参数:
        num_client: 客户端数量
        num_local_class: 每个客户端分配的类别数
        label_adjustment: 每个客户端的标签正确率（0-1）
        k: 类别分配偏移量（控制非IID程度）
    """
    # 设置随机种子（保证可复现性）
    torch.manual_seed(i_seed)
    np.random.seed(i_seed)

    # 加载数据集
    trainset, testset, num_classes = load_data(dataset_name, root='dt2')

    # 参数合法性检查
    assert 0 < num_local_class <= num_classes, "本地类别数必须在1到总类别数之间"
    if label_adjustment is None:
        label_adjustment = [0.0] * num_client
    else:
        assert len(label_adjustment) == num_client, "调整比例数量必须等于客户端数量"
        for ratio in label_adjustment:
            assert 0 <= ratio <= 1, "调整比例必须在0到1之间"

    # 初始化配置变量
    trainset_config = {'users': [], 'user_data': {}, 'num_samples': 0}
    config_division = {}  # 记录每个类别被多少客户端分配
    config_class = {}  # 记录每个客户端分配的类别
    config_data = {}  # 记录每个类别的数据索引

    # 1. 为客户端分配类别（非IID分布）
    for i in range(num_client):
        client_id = f'f_{i:05d}'
        config_class[client_id] = []
        start_cls = i * k  # 起始类别 = 客户端索引 * 偏移量k
        for j in range(num_local_class):
            cls = (start_cls + j) % num_classes  # 循环分配类别
            config_class[client_id].append(cls)
            # 记录每个类别被多少客户端分配
            config_division[cls] = config_division.get(cls, 0) + 1

    # 2. 为每个类别分配数据索引
    for cls in config_division.keys():
        # 获取该类别的所有样本索引
        indexes = torch.nonzero(trainset.original_targets == cls).squeeze()
        if indexes.ndim == 0:  # 处理单样本情况
            indexes = indexes.unsqueeze(0)
        # 随机打乱索引
        indexes = indexes[torch.randperm(len(indexes))]
        # 平均分配给需要该类别的客户端
        num_partition = len(indexes) // config_division[cls]
        config_data[cls] = []
        for i in range(config_division[cls]):
            if i == config_division[cls] - 1:
                # 最后一个客户端分配剩余所有样本
                config_data[cls].append(indexes[i * num_partition:])
            else:
                config_data[cls].append(indexes[i * num_partition: (i + 1) * num_partition])

    # 3. 分配数据给客户端
    total_samples = 0
    class_counter = {cls: 0 for cls in config_division.keys()}  # 记录每个类别的分配进度
    for user in config_class.keys():
        # 收集该客户端的所有样本索引
        user_indices = []
        for cls in config_class[user]:
            user_indices.append(config_data[cls][class_counter[cls]])
            class_counter[cls] += 1
        user_indices = torch.cat(user_indices).tolist()  # 合并索引

        # 初始标签（使用原始标签）
        original_labels = [trainset.original_targets[idx].item() for idx in user_indices]
        user_data = LabelControlledDataset(trainset, user_indices, original_labels)

        # 存入配置
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = user_data
        total_samples += len(user_indices)

    trainset_config['num_samples'] = total_samples
    # 保存原始数据用于对比
    original_client_data = {user: data for user, data in trainset_config['user_data'].items()}

    # 4. 标签错误标注（核心逻辑）
    for user_idx, user in enumerate(config_class.keys()):
        adjustment_ratio = label_adjustment[user_idx]
        if adjustment_ratio <= 0:
            continue  # 不调整的客户端跳过

        user_data = trainset_config['user_data'][user]
        original_labels = user_data.labels.copy()
        new_labels = original_labels.copy()

        # 计算需要错误标注的样本数
        num_to_adjust = int(len(original_labels) * (1 - adjustment_ratio))
        if num_to_adjust > 0:
            # 随机选择需要错误标注的样本
            indices_to_adjust = np.random.choice(
                len(original_labels), num_to_adjust, replace=False
            )
            # 分配错误标签（非原始标签的其他类别）
            for idx in indices_to_adjust:
                true_label = original_labels[idx]
                possible_classes = [c for c in range(num_classes) if c != true_label]
                if possible_classes:
                    new_labels[idx] = np.random.choice(possible_classes)

        # 更新标签
        user_data.labels = new_labels

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


def verify_data_loader(client_data, user_id):
    """验证数据加载器是否使用修改后的标签"""
    data = client_data[user_id]
    dataloader = DataLoader(data, batch_size=5, shuffle=False)

    print(f"\n验证客户端 {user_id} 的数据加载器:")
    # 获取第一个批次的标签
    batch_labels = next(iter(dataloader))[1].tolist()
    # 数据集存储的标签
    expected_labels = data.labels[:5]

    print(f"  数据加载器返回的标签: {batch_labels}")
    print(f"  预期标签: {expected_labels}")

    if batch_labels == expected_labels:
        print("  ✅ 标签匹配：数据加载器使用了修改后的标签")
    else:
        print("  ❌ 标签不匹配：数据加载器仍使用原始标签")


if __name__ == "__main__":
    # 测试非IID分布（k=1）
    print("测试非IID分布（k=1）：")
    train_config, _ = divide_data(
        num_client=5,
        num_local_class=2,
        dataset_name='SelfDataSet',
        i_seed=0,
        print_report=True,
        label_adjustment=[0.0, 0.2, 0.5, 0.8, 1.0],
        k=1
    )

    # 验证数据加载器
    verify_data_loader(train_config['user_data'], 'f_00000')
    verify_data_loader(train_config['user_data'], 'f_00002')

    # 测试更大偏移量（k=2）
    print("\n\n测试非IID分布（k=2）：")
    divide_data(
        num_client=5,
        num_local_class=2,
        dataset_name='SelfDataSet',
        i_seed=0,
        print_report=True,
        label_adjustment=[0.0, 0.0, 0.0, 0.0, 0.0],
        k=2
    )