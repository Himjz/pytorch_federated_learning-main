import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class EfficientDataset(Dataset):
    """高效数据集类：支持GPU加速和多种不可信策略"""

    def __init__(self, base_images, base_labels, indices, custom_labels=None, device=None):
        self.base_images = base_images
        self.base_labels = base_labels
        self.indices = indices
        self.custom_labels = custom_labels if custom_labels is not None else {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noisy_mask = torch.zeros(len(indices), dtype=torch.bool, device='cpu')
        self._precomputed_noise = None

    def set_feature_noise(self, ratio):
        """配置特征噪声策略"""
        num_noisy = max(1, int(len(self) * ratio))
        noisy_indices = np.random.choice(len(self), num_noisy, replace=False)
        self.noisy_mask[noisy_indices] = True

        if num_noisy > 0:
            self._precomputed_noise = {
                'noise_scales': torch.rand(num_noisy, device='cpu') * 0.4 + 0.1,
                'brightness_factors': 1.0 + (torch.rand(num_noisy, device='cpu') - 0.5) * 0.2
            }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        label = self.custom_labels.get(idx, self.base_labels[base_idx])
        image = self.base_images[base_idx].to(self.device)

        if self.noisy_mask[idx]:
            noise_idx = (self.noisy_mask[:idx]).sum().item()
            noise_scale = self._precomputed_noise['noise_scales'][noise_idx].to(self.device)
            brightness_factor = self._precomputed_noise['brightness_factors'][noise_idx].to(self.device)
            image = image + torch.randn_like(image, device=self.device) * noise_scale
            image = torch.clamp(image * brightness_factor, 0.0, 1.0)

        return image, label


def to_numpy_label(label):
    """将标签转换为普通整数"""
    return label.item() if torch.is_tensor(label) else label


def load_data(name, root='./data', download=True, load_path=None, device=None, preload_to_gpu=False):
    """加载数据集，支持默认数据集和自定义数据集"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    default_datasets = {
        'MNIST': (torchvision.datasets.MNIST, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]), 10),
        'CIFAR10': (torchvision.datasets.CIFAR10, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ]), 10),
        'FashionMNIST': (torchvision.datasets.FashionMNIST, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), 10),
        'CIFAR100': (torchvision.datasets.CIFAR100, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]), 100),
    }

    # 处理默认数据集
    if name in default_datasets:
        dataset_cls, transform, num_classes = default_datasets[name]
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

        trainset = dataset_cls(root=root, train=True, download=download, transform=transform)
        testset = dataset_cls(root=root, train=False, download=download, transform=transform)

        # 确保标签为张量
        if hasattr(trainset, 'targets'):
            trainset.targets = torch.Tensor(trainset.targets)
            testset.targets = torch.Tensor(testset.targets)
        elif hasattr(trainset, 'labels'):
            trainset.targets = torch.Tensor(trainset.labels)
            testset.targets = torch.Tensor(testset.labels)

        base_images = [trainset[i][0].to(device) for i in range(len(trainset))] if preload_to_gpu else [
            trainset[i][0] for i in range(len(trainset))]
        base_labels = trainset.targets

        return base_images, base_labels, num_classes, testset

    # 处理自定义数据集
    else:
        # 从root目录下寻找与dataset_name同名的目录
        dataset_root = os.path.join(root, name)

        # 检查目录是否存在
        if not os.path.exists(dataset_root):
            raise ValueError(f"数据集目录不存在: {dataset_root}")

        # 验证目录结构是否符合要求
        train_dir = os.path.join(dataset_root, 'train')
        val_dir = os.path.join(dataset_root, 'val')

        # 检查train和val子目录是否存在
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise ValueError(f"数据集目录结构不符合要求。需要包含'train'和'val'子目录: {dataset_root}")

        # 检查train和val目录下是否有类别子目录
        train_classes = os.listdir(train_dir)
        val_classes = os.listdir(val_dir)

        if len(train_classes) == 0:
            raise ValueError(f"'train'目录为空: {train_dir}")

        if len(val_classes) == 0:
            raise ValueError(f"'val'目录为空: {val_dir}")

        # 检查类别是否一致（可选）
        if set(train_classes) != set(val_classes):
            print(f"警告: 'train'和'val'目录下的类别不完全一致: {train_classes} vs {val_classes}")

        # 确定图像通道数（根据第一个图像）
        sample_img_path = os.path.join(train_dir, train_classes[0],
                                       os.listdir(os.path.join(train_dir, train_classes[0]))[0])
        sample_img = Image.open(sample_img_path)
        num_channels = len(sample_img.getbands())

        # 设置相应的转换
        if num_channels == 1:  # 灰度图像
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.229])
            ])
        else:  # RGB图像
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # 加载数据集
        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
        testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)

        base_images = [trainset[i][0].to(device) for i in range(len(trainset))] if preload_to_gpu else [
            trainset[i][0] for i in range(len(trainset))]
        base_labels = torch.Tensor(trainset.targets)
        num_classes = len(trainset.classes)

        print(f"成功加载自定义数据集: {dataset_root}")
        print(f"训练集样本数: {len(trainset)}, 验证集样本数: {len(testset)}")
        print(f"类别数: {num_classes}, 类别列表: {trainset.classes}")

        return base_images, base_labels, num_classes, testset


def parse_strategy(strategy_code):
    """解析策略代码（如2.1→稀疏，3.5→特征噪声）"""
    try:
        code = float(strategy_code)
        integer_part = int(code)
        decimal_part = code - integer_part

        strategies = {
            0: ('malicious', decimal_part),  # 恶意（准确率x）
            2: ('sparse', decimal_part),  # 数据稀疏（比例x）
            3: ('feature_noise', decimal_part),  # 特征噪声（比例x）
            4: ('class_skew', decimal_part),  # 类别倾斜（比例x）
        }

        return strategies.get(integer_part, ('normal', 1.0))
    except (ValueError, TypeError):
        return 'normal', 1.0


def apply_strategy(ds, strategy, ratio, client_classes, client_id):
    """应用不可信策略到数据集"""
    if strategy == 'sparse':
        ds.is_data_sparse = True
    elif strategy == 'feature_noise':
        ds.set_feature_noise(ratio)
    elif strategy == 'class_skew':
        ds.is_class_skewed = True
    elif strategy == 'malicious':
        local_classes = client_classes[client_id]
        num_corrupt = max(1, int(len(ds) * (1 - ratio)))

        for idx in np.random.choice(len(ds), num_corrupt, replace=False):
            true_label = to_numpy_label(ds.base_labels[ds.indices[idx]])
            if true_label in local_classes:
                possible_labels = [c for c in local_classes if c != true_label]
                if possible_labels:
                    ds.custom_labels[idx] = np.random.choice(possible_labels)


def print_distribution(clients, num_classes, title):
    """打印数据集分布"""
    print(f"\n===== {title} =====")
    for client_id, ds in clients.items():
        labels = [ds.custom_labels.get(idx, ds.base_labels[ds.indices[idx]]) for idx in range(len(ds))]
        counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
        print(f"客户端 {client_id}:")
        for cls in range(num_classes):
            if counts[cls] > 0:
                print(f"  类别 {cls}: {counts[cls]} 个样本")
        if hasattr(ds, 'is_data_sparse'):
            print(f"  数据稀疏: 是" if ds.is_data_sparse else "  数据稀疏: 否")
        if ds.noisy_mask.any():
            print(f"  特征噪声样本数: {ds.noisy_mask.sum().item()}")
        if hasattr(ds, 'is_class_skewed'):
            print(f"  类别倾斜: 是" if ds.is_class_skewed else "  类别倾斜: 否")
        print()


def calculate_accuracy(original_clients, adjusted_clients, device):
    """计算标签准确率"""
    accuracies = {}
    for client_id in original_clients:
        orig_ds = original_clients[client_id]
        adj_ds = adjusted_clients[client_id]

        # 获取调整后样本的原始索引
        adjusted_orig_indices = [orig_ds.indices[idx] for idx in range(len(adj_ds))]

        # 获取原始标签和调整后的标签
        orig_labels = torch.tensor([orig_ds.base_labels[idx] for idx in adjusted_orig_indices], device=device)
        adj_labels = torch.tensor(
            [adj_ds.custom_labels.get(idx, adj_ds.base_labels[adj_ds.indices[idx]]) for idx in range(len(adj_ds))],
            device=device)

        # 计算准确率
        accuracies[client_id] = (orig_labels == adj_labels).sum().item() / len(orig_labels) if len(
            orig_labels) > 0 else 0.0

    return accuracies


def divide_data(
        num_client=1,
        num_local_class=10,
        dataset_name='emnist',
        i_seed=0,
        untrusted_strategies=None,
        k=1,
        print_report=True,
        device=None,
        preload_to_gpu=False,
        load_path=None
):
    """划分数据集并应用不可信策略"""
    torch.manual_seed(i_seed)
    np.random.seed(i_seed)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    base_images, base_labels, num_classes, testset = load_data(
        name=dataset_name,
        load_path=load_path,
        device=device,
        preload_to_gpu=preload_to_gpu
    )

    # 按类别分组索引
    class_indices = {cls: [] for cls in range(num_classes)}
    for idx, cls in enumerate(base_labels):
        class_indices[to_numpy_label(cls)].append(idx)
    for indices in class_indices.values():
        np.random.shuffle(indices)

    # 客户端-类别分配
    client_classes = {f'f_{i:05d}': [(i * k + j) % num_classes for j in range(num_local_class)] for i in
                      range(num_client)}
    class_assign_counts = {cls: 0 for cls in range(num_classes)}
    for classes in client_classes.values():
        for cls in classes:
            class_assign_counts[cls] += 1

    # 分配样本索引
    client_indices = {cid: [] for cid in client_classes}
    for client_id, classes in client_classes.items():
        # 获取客户端ID对应的索引
        client_index = int(client_id.split('_')[-1])

        # 获取策略代码
        strategy_code = untrusted_strategies[client_index] if untrusted_strategies and client_index < len(
            untrusted_strategies) else 1.0
        strategy, ratio = parse_strategy(strategy_code)

        # 类别倾斜策略需要选择主导类别
        dominant_cls = random.choice(classes) if strategy == 'class_skew' else None

        for cls in classes:
            if class_assign_counts[cls] <= 0:
                continue

            # 基础分配数量
            per_client = len(class_indices[cls]) // class_assign_counts[cls] if class_assign_counts[cls] > 1 else len(
                class_indices[cls])

            # 应用策略调整样本数量
            if strategy == 'sparse':
                per_client = max(1, int(per_client * ratio))
            elif strategy == 'class_skew':
                per_client = per_client * 2 if cls == dominant_cls else max(1, int(per_client * (1 - ratio)))

            # 分配样本
            client_indices[client_id].extend(class_indices[cls][:per_client])
            class_indices[cls] = class_indices[cls][per_client:]
            class_assign_counts[cls] -= 1

    # 创建数据集
    clients = {}
    original_clients = {}
    for client_id in client_classes:
        indices = client_indices[client_id]
        clients[client_id] = EfficientDataset(base_images, base_labels, indices, device=device)
        original_clients[client_id] = EfficientDataset(base_images, base_labels, indices, device=device)

    # 应用策略
    if untrusted_strategies:
        for client_id in clients:
            client_index = int(client_id.split('_')[-1])
            if client_index >= len(untrusted_strategies):
                continue

            strategy_code = untrusted_strategies[client_index]
            strategy, ratio = parse_strategy(strategy_code)
            apply_strategy(clients[client_id], strategy, ratio, client_classes, client_id)

    # 封装结果
    trainset_config = {
        'users': list(clients.keys()),
        'user_data': clients,
        'num_samples': sum(len(ds) for ds in clients.values())
    }

    # 打印报告
    if print_report:
        print(f"===== 数据划分（设备: {device}） =====")
        if untrusted_strategies:
            print("===== 不可信策略分配 =====")
            for client_id in clients:
                client_index = int(client_id.split('_')[-1])
                strategy_code = untrusted_strategies[client_index] if client_index < len(untrusted_strategies) else 1.0
                strategy, ratio = parse_strategy(strategy_code)
                print(f"客户端 {client_id}: 策略 = {strategy}, 参数 = {ratio:.2f}")

        print_distribution(original_clients, num_classes, "原始分布")
        print_distribution(clients, num_classes, "应用策略后分布")

        if untrusted_strategies:
            accuracies = calculate_accuracy(original_clients, clients, device)
            print("\n===== 标签准确率报告 =====")
            for client_id, acc in accuracies.items():
                print(f"客户端 {client_id}: 准确率 = {acc:.4f}")
            print(f"平均准确率: {sum(accuracies.values()) / len(accuracies):.4f}" if accuracies else "无准确率数据")

    return trainset_config, testset


def verify_loader(client_data, client_id, device):
    """验证数据加载器"""
    ds = client_data[client_id]
    batch_size = min(5, len(ds))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cpu'),
        num_workers=0 if device.type == 'cuda' else 2
    )

    try:
        batch_imgs, batch_labels = next(iter(loader))
    except StopIteration:
        print(f"\n验证客户端 {client_id} 的加载器:")
        print("  ❌ 数据集为空，无法获取批次数据")
        return

    print(f"\n验证客户端 {client_id} 的加载器:")
    print(f"  图像设备: {batch_imgs.device}")
    print(f"  返回标签: {batch_labels.tolist()}")
    print(f"  预期标签: {[ds.custom_labels.get(idx, ds.base_labels[ds.indices[idx]]) for idx in range(batch_size)]}")
    if ds.noisy_mask.any():
        print(f"  批量中含特征噪声的样本索引: {[idx for idx in range(batch_size) if ds.noisy_mask[idx].item()]}")
    print("  ✅ 验证通过" if (
                batch_labels.tolist() == [ds.custom_labels.get(idx, ds.base_labels[ds.indices[idx]]) for idx in
                                          range(batch_size)]) else "  ❌ 验证失败")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    trainset_config, testset = divide_data(
        num_client=5,
        num_local_class=2,
        dataset_name='MNIST',
        i_seed=0,
        untrusted_strategies=[2.1, 3.5, 4.3, 0.5, 1.0],
        device=device,
        preload_to_gpu=False,
        load_path=None
    )

    for i in range(5):
        verify_loader(trainset_config['user_data'], f'f_0000{i}', device)