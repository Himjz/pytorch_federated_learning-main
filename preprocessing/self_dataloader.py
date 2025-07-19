import os
from typing import List

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class EfficientDataset(Dataset):
    """高效数据集类：支持GPU加速的特征噪声生成"""

    def __init__(self, base_images, base_labels, indices, custom_labels=None, noisy_mask=None, device=None):
        # 存储原始数据（CPU，避免冗余GPU内存占用）
        self.base_images = base_images  # list of tensors (CPU)
        self.base_labels = base_labels  # list of integers
        self.indices = indices  # 样本索引（指向base_images）
        self.custom_labels = custom_labels if custom_labels is not None else {}

        # 特征噪声掩码（布尔数组，标记哪些样本需要加噪声）
        self.noisy_mask = noisy_mask if noisy_mask is not None else torch.zeros(len(indices), dtype=torch.bool)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 预计算噪声参数（仅在有噪声样本时）
        if self.noisy_mask.any():
            self._precompute_noise_params()

    def __len__(self):
        return len(self.indices)

    def _precompute_noise_params(self):
        """预计算噪声参数（GPU加速）"""
        num_noisy = self.noisy_mask.sum().item()
        # 高斯噪声标准差（与策略参数相关，这里预生成0.1-0.5的随机值）
        self.noise_scales = torch.rand(num_noisy, device=self.device) * 0.4 + 0.1
        # 亮度调整因子
        self.brightness_factors = 1.0 + (torch.rand(num_noisy, device=self.device) - 0.5) * 0.2

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        label = self.custom_labels.get(idx, self.base_labels[base_idx])
        image = self.base_images[base_idx].to(self.device)  # 动态转移到GPU

        # 对标记为噪声的样本添加特征噪声（GPU上执行）
        if self.noisy_mask[idx]:
            # 查找当前噪声样本在预计算参数中的索引
            noise_idx = self.noisy_mask[:idx].sum().item()
            # 高斯噪声
            gaussian_noise = torch.randn_like(image, device=self.device) * self.noise_scales[noise_idx]
            image = image + gaussian_noise
            # 亮度调整
            image = torch.clamp(image * self.brightness_factors[noise_idx], 0.0, 1.0)

        return image, label


def load_base_data(root='../dt2', device=None):
    """加载原始数据并预处理（支持GPU预加载）"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # 加载数据集
    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(root, 'train'),
        transform=transform
    )
    testset = torchvision.datasets.ImageFolder(
        root=os.path.join(root, 'val'),
        transform=transform
    )

    # 预加载图像到GPU（如果内存允许）
    base_images = []
    for i in range(len(trainset)):
        img = trainset[i][0].to(device, non_blocking=True)  # 非阻塞转移
        base_images.append(img)
    base_labels = trainset.targets
    num_classes = len(trainset.classes)

    return base_images, base_labels, num_classes, testset


def print_distribution(clients, num_classes, title):
    """高效打印分布（批量统计，减少IO操作）"""
    print(f"\n===== {title} =====")
    # 批量收集所有客户端的统计信息
    stats = []
    for client_id, ds in clients.items():
        labels = [ds.custom_labels.get(idx, ds.base_labels[ds.indices[idx]]) for idx in range(len(ds))]
        counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
        stats.append((client_id, counts))

    # 统一打印
    for client_id, counts in stats:
        print(f"客户端 {client_id}:")
        for cls in range(num_classes):
            if counts[cls] > 0:
                print(f"  类别 {cls}: {counts[cls]} 个样本")
        if hasattr(ds, 'noisy_mask') and ds.noisy_mask.any():
            print(f"  特征噪声样本数: {ds.noisy_mask.sum().item()}")
        if hasattr(ds, 'is_data_sparse') and ds.is_data_sparse:
            print(f"  数据稀疏性: 样本数少（{len(ds)}个样本）")
        if hasattr(ds, 'is_class_skewed') and ds.is_class_skewed:
            print(f"  类别倾斜: 主导类别 = {counts.argmax().item()}")
        print()


def calculate_accuracy(original_clients, adjusted_clients, device):
    """批量计算准确率（适配样本数变化）"""
    accuracies = {}
    for client_id in original_clients:
        orig_ds = original_clients[client_id]
        adj_ds = adjusted_clients[client_id]

        # 仅对调整后保留的样本计算准确率
        # 1. 获取调整后样本的原始索引（adj_ds.indices是调整后的索引）
        adjusted_original_indices = [orig_ds.indices[idx] for idx in range(len(adj_ds))]
        # 2. 提取这些样本在原始数据中的标签
        orig_labels = torch.tensor(
            [orig_ds.base_labels[orig_idx] for orig_idx in adjusted_original_indices],
            device=device
        )
        # 3. 提取调整后的标签
        adj_labels = torch.tensor(
            [adj_ds.custom_labels.get(idx, adj_ds.base_labels[adj_ds.indices[idx]]) for idx in range(len(adj_ds))],
            device=device
        )

        # 计算准确率（确保长度一致）
        correct = (orig_labels == adj_labels).sum().item()
        accuracies[client_id] = correct / len(orig_labels) if len(orig_labels) > 0 else 0.0
    return accuracies


def parse_strategy(strategy_code):
    """解析策略代码（如2.1、3.5）"""
    strategy_code = float(strategy_code)
    integer_part = int(strategy_code)
    decimal_part = strategy_code - integer_part

    if integer_part == 0:
        return 'malicious', decimal_part  # 恶意不可信（随机标签）
    elif integer_part == 2:
        return 'sparse', decimal_part  # 数据稀疏
    elif integer_part == 3:
        return 'feature_noise', decimal_part  # 特征噪声
    elif integer_part == 4:
        return 'class_skew', decimal_part  # 类别倾斜
    else:
        return 'normal', 1.0  # 正常客户端


def divide_data(
        num_client=5,
        num_local_class=2,
        untrusted_strategies:List[float]=None,  # 浮点型数组，如[2.1,3.5,4.3,0.5,1.0]
        k=1,
        print_report=True,
        device=None
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 解析浮点型数组为客户端-策略映射
    if isinstance(untrusted_strategies, list) and all(isinstance(x, (int, float)) for x in untrusted_strategies):
        strategy_map = {f'f_{i:05d}': untrusted_strategies[i] for i in
                        range(min(num_client, len(untrusted_strategies)))}
        untrusted_strategies = strategy_map
    elif untrusted_strategies is None:
        untrusted_strategies = {f'f_{i:05d}': 1.0 for i in range(num_client)}  # 默认全正常

    # 1. 加载原始数据（GPU加速）
    base_images, base_labels, num_classes, testset = load_base_data(device=device)
    np.random.seed(0)
    torch.manual_seed(0)

    # 2. 按类别预分组索引（批量操作）
    class_indices = {cls: [] for cls in range(num_classes)}
    for idx, cls in enumerate(base_labels):
        class_indices[cls].append(idx)
    # 批量打乱
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    # 3. 客户端-类别分配
    client_classes = {f'f_{i:05d}': [(i * k + j) % num_classes for j in range(num_local_class)] for i in
                      range(num_client)}

    # 4. 预计算类别分配次数
    class_assign_counts = {cls: 0 for cls in range(num_classes)}
    for classes in client_classes.values():
        for cls in classes:
            class_assign_counts[cls] += 1

    # 5. 分配样本索引（批量处理）
    client_indices = {cid: [] for cid in client_classes}
    for client_id, classes in client_classes.items():
        strategy_code = untrusted_strategies.get(client_id, 1.0)
        strategy, ratio = parse_strategy(strategy_code)

        for cls in classes:
            total = len(class_indices[cls])
            if class_assign_counts[cls] == 0:
                continue
            per_client = total // class_assign_counts[cls] if class_assign_counts[cls] > 1 else total
            # 数据稀疏客户端：减少样本数
            if strategy == 'sparse':
                per_client = max(1, int(per_client * ratio))
            client_indices[client_id].extend(class_indices[cls][:per_client])
            class_indices[cls] = class_indices[cls][per_client:]
            class_assign_counts[cls] -= 1

    # 6. 创建客户端数据集（GPU优化）
    clients = {}
    original_clients = {}
    for client_id in client_classes:
        indices = client_indices[client_id]
        # 原始数据集（无策略）
        original_clients[client_id] = EfficientDataset(
            base_images, base_labels, indices, device=device
        )
        # 应用策略的数据集
        clients[client_id] = EfficientDataset(
            base_images, base_labels, indices, device=device
        )

    # 7. 批量应用不可信策略（GPU加速）
    for client_id, strategy_code in untrusted_strategies.items():
        strategy, ratio = parse_strategy(strategy_code)
        ds = clients.get(client_id)
        if not ds:
            continue
        total = len(ds)
        if total == 0:
            continue

        if strategy == 'sparse':
            ds.is_data_sparse = True

        elif strategy == 'feature_noise':
            # 生成噪声掩码（布尔数组，GPU存储）
            num_noisy = max(1, int(total * ratio))
            noisy_indices = np.random.choice(total, num_noisy, replace=False)
            noisy_mask = torch.zeros(total, dtype=torch.bool, device=device)
            noisy_mask[noisy_indices] = True
            ds.noisy_mask = noisy_mask
            ds._precompute_noise_params()  # 预计算噪声参数

        elif strategy == 'class_skew':
            # 类别倾斜（批量筛选）
            dominant_cls = random.choice(range(num_classes))
            # 批量获取所有样本的原始标签
            all_labels = torch.tensor([ds.base_labels[ds.indices[idx]] for idx in range(total)])
            # 筛选主导类别和非主导类别索引
            dominant_mask = (all_labels == dominant_cls)
            non_dominant_indices = torch.where(~dominant_mask)[0].numpy()
            # 保留部分非主导类别样本
            keep_ratio = 1.0 - ratio
            keep_size = max(1, int(len(non_dominant_indices) * keep_ratio))
            keep_indices = np.random.choice(non_dominant_indices, keep_size,
                                            replace=False) if non_dominant_indices.size else []
            # 合并索引
            new_indices = np.concatenate([
                np.where(dominant_mask.numpy())[0],
                keep_indices
            ])
            # 更新索引
            ds.indices = [ds.indices[idx] for idx in new_indices]
            ds.is_class_skewed = True

        elif strategy == 'malicious':
            # 恶意标签（批量生成）
            num_corrupt = max(1, int(total * (1 - ratio)))
            corrupt_indices = np.random.choice(total, num_corrupt, replace=False)
            # 批量获取真实标签
            true_labels = [ds.base_labels[ds.indices[idx]] for idx in corrupt_indices]
            # 批量生成错误标签
            possible_labels = []
            for tl in true_labels:
                possible = [c for c in range(num_classes) if c != tl]
                possible_labels.append(np.random.choice(possible) if possible else tl)
            # 批量更新标签
            for idx, new_label in zip(corrupt_indices, possible_labels):
                ds.custom_labels[idx] = new_label

    # 8. 封装训练集配置
    trainset_config = {
        'users': list(clients.keys()),
        'user_data': clients,
        'num_samples': sum(len(ds) for ds in clients.values())
    }

    # 9. 打印报告（控制频率，减少IO耗时）
    if print_report:
        print(f"===== 不可信策略分配（设备: {device}） =====")
        for cid, code in untrusted_strategies.items():
            strategy, ratio = parse_strategy(code)
            print(f"客户端 {cid}: 策略 = {strategy}, 参数 = {ratio:.2f}")
        print_distribution(original_clients, num_classes, "原始分布")
        print_distribution(clients, num_classes, "应用策略后分布")
        # 批量计算准确率
        accuracies = calculate_accuracy(original_clients, clients, device=device)
        print("\n===== 标签准确率报告 =====")
        avg_acc = sum(accuracies.values()) / len(accuracies) if accuracies else 0.0
        for cid, acc in accuracies.items():
            print(f"客户端 {cid}: 准确率 = {acc:.4f}")
        print(f"平均准确率: {avg_acc:.4f}")

    return trainset_config, testset


def verify_loader(client_data, client_id, device):
    """验证数据加载器（修复pin_memory冲突）"""
    ds = client_data[client_id]
    # 若数据已在GPU，关闭pin_memory（仅CPU数据需要）
    pin_memory = (device.type == 'cpu')  # 仅CPU时启用pin_memory
    loader = DataLoader(ds, batch_size=5, shuffle=False, pin_memory=pin_memory)
    batch_imgs, batch_labels = next(iter(loader))
    print(f"\n验证客户端 {client_id} 的加载器:")
    print(f"  图像设备: {batch_imgs.device}")
    print(f"  返回标签: {batch_labels.tolist()}")
    print(f"  预期标签: {[ds.custom_labels.get(idx, ds.base_labels[ds.indices[idx]]) for idx in range(5)]}")
    # 验证特征噪声
    if hasattr(ds, 'noisy_mask') and ds.noisy_mask.any():
        noisy_in_batch = [idx for idx in range(5) if ds.noisy_mask[idx].item()]
        print(f"  批量中含特征噪声的样本索引: {noisy_in_batch}")
    print("  ✅ 验证通过" if (batch_labels.tolist() == [ds.custom_labels.get(idx, ds.base_labels[ds.indices[idx]]) for idx in range(5)]) else "  ❌ 验证失败")


if __name__ == "__main__":
    # 测试GPU加速版本
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    # 传入浮点型数组配置策略
    trainset_config, testset = divide_data(
        num_client=5,
        untrusted_strategies=[2.1, 3.5, 4.3, 0.5, 1.0],
        device=device
    )
    # 验证各客户端
    for i in range(5):
        verify_loader(trainset_config['user_data'], f'f_0000{i}', device)