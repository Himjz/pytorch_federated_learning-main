import random

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils import *


class EfficientDataset(Dataset):
    """高效数据集类：支持GPU加速和多种不可信策略"""
    def __init__(self, base_images, base_labels, indices, custom_labels=None, device=None):
        self.base_images = base_images
        self.base_labels = base_labels
        self.indices = indices
        self.custom_labels = custom_labels or {}
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


class DataSetInfo:
    """数据集信息类，用于存储数据集的元信息"""
    def __init__(self, num_classes, image_dim, image_channel):
        self.num_classes = num_classes
        self.image_dim = image_dim
        self.image_channel = image_channel

    def __str__(self):
        return (f"数据集信息:\n"
                f"  类别数: {self.num_classes}\n"
                f"  图像尺寸: {self.image_dim}x{self.image_dim}\n"
                f"  图像通道数: {self.image_channel}")

    def get(self):
        """返回数据集的基本信息"""
        return self.num_classes, self.image_dim, self.image_channel


def load_data(name, root='./data', download=True, device=None, preload_to_gpu=False, in_channels=3, image_size=None):
    """加载数据集，支持默认数据集和自定义数据集"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集配置映射
    dataset_configs = {
        'MNIST': (torchvision.datasets.MNIST, 10, 'targets', 1, 28),
        'CIFAR10': (torchvision.datasets.CIFAR10, 10, 'targets', 3, 32),
        'FashionMNIST': (torchvision.datasets.FashionMNIST, 10, 'targets', 1, 28),
        'CIFAR100': (torchvision.datasets.CIFAR100, 100, 'targets', 3, 32),
        'EMNIST': (torchvision.datasets.EMNIST, 47, 'targets', 1, 28),
        'SVHN': (torchvision.datasets.SVHN, 10, 'labels', 3, 32),
    }

    # 处理默认数据集
    if name in dataset_configs:
        dataset_cls, num_classes, label_attr, default_channels, default_size = dataset_configs[name]
        os.makedirs(root, exist_ok=True)

        # 确定图像尺寸
        if image_size is None:
            image_size = default_size
            print(f"使用默认图像尺寸 {image_size}x{image_size} 用于 {name} 数据集")
        else:
            print(f"使用指定图像尺寸 {image_size}x{image_size} 用于 {name} 数据集")

        # 创建transform
        transform = create_transform(image_size, in_channels or default_channels)

        # 特殊处理不同数据集的加载参数
        if name == 'EMNIST':
            trainset = dataset_cls(root=root, split='balanced', train=True, download=download, transform=transform)
            testset = dataset_cls(root=root, split='balanced', train=False, download=download, transform=transform)
        elif name == 'SVHN':
            trainset = dataset_cls(root=root, split='train', download=download, transform=transform)
            testset = dataset_cls(root=root, split='test', download=download, transform=transform)
        else:
            trainset = dataset_cls(root=root, train=True, download=download, transform=transform)
            testset = dataset_cls(root=root, train=False, download=download, transform=transform)

        # 统一标签处理
        base_labels = getattr(trainset, label_attr)
        if not torch.is_tensor(base_labels):
            base_labels = torch.tensor(base_labels)
        test_labels = getattr(testset, label_attr)
        if not torch.is_tensor(test_labels):
            test_labels = torch.tensor(test_labels)
        setattr(trainset, 'targets', base_labels)
        setattr(testset, 'targets', test_labels)

        # 预加载图像到GPU（如果需要）
        base_images = [trainset[i][0].to(device) for i in range(len(trainset))] if preload_to_gpu else [
            trainset[i][0] for i in range(len(trainset))]

        return base_images, base_labels, num_classes, testset, DataSetInfo(num_classes, image_size, in_channels or default_channels)

    # 处理自定义数据集
    else:
        dataset_root = os.path.join(root, name)
        if not os.path.exists(dataset_root):
            raise ValueError(f"数据集目录不存在: {dataset_root}")

        train_dir = os.path.join(dataset_root, 'train')
        val_dir = os.path.join(dataset_root, 'val')
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise ValueError(f"数据集目录结构不符合要求。需要包含'train'和'val'子目录: {dataset_root}")

        # 确定图像尺寸
        if image_size is None:
            print("检测数据集中的主要图像尺寸...")
            image_size = detect_dominant_image_size(train_dir)
            print(f"检测到主要图像尺寸: {image_size}x{image_size}")
        else:
            print(f"使用指定图像尺寸 {image_size}x{image_size}")

        # 创建transform
        transform = create_transform(image_size, in_channels)

        # 加载数据集
        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
        testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)

        base_images = [trainset[i][0].to(device) for i in range(len(trainset))] if preload_to_gpu else [
            trainset[i][0] for i in range(len(trainset))]
        base_labels = torch.tensor(trainset.targets)
        num_classes = len(trainset.classes)
        image_channel = in_channels or (3 if len(trainset[0][0]) == 3 else 1)

        print(f"成功加载自定义数据集: {dataset_root}")
        print(f"训练集样本数: {len(trainset)}, 验证集样本数: {len(testset)}")
        print(f"类别数: {num_classes}, 类别列表: {trainset.classes}")

        return base_images, base_labels, num_classes, testset, DataSetInfo(num_classes, image_size, image_channel)


def create_transform(image_size, in_channels):
    """创建图像转换"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1) if in_channels == 1 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5] if in_channels == 1 else [0.485, 0.456, 0.406],
            std=[0.229] if in_channels == 1 else [0.229, 0.224, 0.225]
        )
    ])


def divide_data(
        num_client=1,
        num_local_class=10,
        dataset_name='MNIST',
        i_seed=0,
        untrusted_strategies=None,
        k=1,
        print_report=True,
        device=None,
        preload_to_gpu=False,
        in_channels=1,
        image_size=None
):
    """划分数据集并应用不可信策略"""
    torch.manual_seed(i_seed)
    np.random.seed(i_seed)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    base_images, base_labels, num_classes, testset, dataset_info = load_data(
        name=dataset_name,
        device=device,
        preload_to_gpu=preload_to_gpu,
        in_channels=in_channels,
        image_size=image_size
    )

    if print_report:
        print(dataset_info)

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
        client_index = int(client_id.split('_')[-1])
        strategy_code = untrusted_strategies[client_index] if untrusted_strategies and client_index < len(
            untrusted_strategies) else 1.0
        strategy, ratio = parse_strategy(strategy_code)
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

    return trainset_config, testset, dataset_info


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
    print(f"  图像形状: {batch_imgs.shape}")
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

    # 测试MNIST数据集
    trainset_config, testset, dataset_info = divide_data(
        num_client=5,
        num_local_class=2,
        dataset_name='MNIST',
        i_seed=0,
        untrusted_strategies=[2.1, 3.5, 4.3, 0.5, 1.0],
        device=device,
        preload_to_gpu=False,
        in_channels=1,
        image_size=32  # 指定图像尺寸
    )

    print("\n数据集信息:")
    num_classes, img_dim, img_channel = dataset_info.get()
    print(f"  类别数: {num_classes}")
    print(f"  图像尺寸: {img_dim}x{img_dim}")
    print(f"  图像通道数: {img_channel}")

    for i in range(5):
        verify_loader(trainset_config['user_data'], f'f_0000{i}', device)