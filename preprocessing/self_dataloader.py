import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EfficientDataset(Dataset):
    """高效数据集类：通过索引引用原始数据，避免复制"""
    def __init__(self, base_images, base_labels, indices, custom_labels=None):
        self.base_images = base_images  # 原始图像列表（引用，不复制）
        self.base_labels = base_labels  # 原始标签列表（引用）
        self.indices = indices  # 样本索引（指向原始数据）
        # 自定义标签：默认使用原始标签，修改时仅存储变化的标签
        self.custom_labels = custom_labels if custom_labels is not None else {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 原始数据索引
        base_idx = self.indices[idx]
        # 优先使用自定义标签，否则用原始标签
        label = self.custom_labels.get(idx, self.base_labels[base_idx])
        return self.base_images[base_idx], label


def load_base_data(root='dt2'):
    """加载原始数据（仅加载一次，返回引用）"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # 加载数据集（仅加载路径和标签，不加载图像到内存）
    dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(root, 'train'),
        transform=transform
    )
    # 预加载所有图像（一次性加载，后续通过索引访问）
    images = [dataset[i][0] for i in range(len(dataset))]
    labels = [dataset.targets[i] for i in range(len(dataset))]
    return images, labels, len(dataset.classes)


def print_distribution(clients, num_classes, title):
    """高效打印分布：批量统计标签"""
    print(f"\n===== {title} =====")
    for client_id, ds in clients.items():
        # 批量获取标签（仅计算需要的标签）
        labels = []
        for idx in range(len(ds)):
            base_idx = ds.indices[idx]
            labels.append(ds.custom_labels.get(idx, ds.base_labels[base_idx]))
        counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
        print(f"客户端 {client_id}:")
        for cls in range(num_classes):
            if counts[cls] > 0:
                print(f"  类别 {cls}: {counts[cls]} 个样本")
        print()


def calculate_accuracy(original_clients, adjusted_clients):
    """批量计算准确率"""
    accuracies = {}
    for client_id in original_clients:
        orig_ds = original_clients[client_id]
        adj_ds = adjusted_clients[client_id]
        # 批量获取原始标签和调整后标签
        orig_labels = [orig_ds.base_labels[orig_ds.indices[idx]] for idx in range(len(orig_ds))]
        adj_labels = [adj_ds.custom_labels.get(idx, adj_ds.base_labels[adj_ds.indices[idx]])
                     for idx in range(len(adj_ds))]
        # 计算准确率
        correct = np.sum(np.array(orig_labels) == np.array(adj_labels))
        accuracies[client_id] = correct / len(orig_labels) if orig_labels else 0.0
    return accuracies


def divide_data(num_client=5, num_local_class=2, target_accuracies=None, k=1, print_report=True):
    # 1. 加载原始数据（仅一次）
    base_images, base_labels, num_classes = load_base_data()
    np.random.seed(0)  # 固定随机种子，保证可复现

    # 2. 按类别预分组索引（高效分组）
    class_indices = {cls: [] for cls in range(num_classes)}
    for idx, cls in enumerate(base_labels):
        class_indices[cls].append(idx)
    # 打乱每个类别的索引（原地操作，无复制）
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    # 3. 计算客户端-类别分配（预计算，避免重复）
    client_classes = {}  # {client_id: [分配的类别]}
    for i in range(num_client):
        client_id = f'f_{i:05d}'
        start_cls = (i * k) % num_classes
        client_classes[client_id] = [(start_cls + j) % num_classes for j in range(num_local_class)]

    # 4. 预计算每个类别的分配次数（用于样本分配）
    class_assign_counts = {cls: 0 for cls in range(num_classes)}
    for classes in client_classes.values():
        for cls in classes:
            class_assign_counts[cls] += 1

    # 5. 分配样本索引（仅操作索引，不复制数据）
    client_indices = {cid: [] for cid in client_classes}  # 存储每个客户端的样本索引
    for client_id, classes in client_classes.items():
        for cls in classes:
            total = len(class_indices[cls])
            if class_assign_counts[cls] == 0:
                continue
            # 计算分配数量（整数除法，高效）
            per_client = total // class_assign_counts[cls]
            if class_assign_counts[cls] == 1:
                per_client = total  # 最后一个客户端分配剩余样本
            # 分配索引（切片操作高效）
            client_indices[client_id].extend(class_indices[cls][:per_client])
            class_indices[cls] = class_indices[cls][per_client:]  # 剩余索引
            class_assign_counts[cls] -= 1

    # 6. 创建客户端数据集（高效引用原始数据）
    clients = {}
    original_clients = {}  # 存储原始标签状态（仅索引，无数据复制）
    for client_id in client_classes:
        indices = client_indices[client_id]
        # 原始数据集（无标签修改）
        original_clients[client_id] = EfficientDataset(base_images, base_labels, indices)
        # 可修改标签的数据集
        clients[client_id] = EfficientDataset(base_images, base_labels, indices)

    # 7. 批量修改标签（NumPy加速）
    for i, (client_id, ds) in enumerate(clients.items()):
        target_acc = target_accuracies[i]
        total = len(ds)
        if total == 0:
            continue

        # 计算需要修改的样本数（向量化计算）
        num_corrupt = int(round(total * (1 - target_acc)))
        num_corrupt = max(1, num_corrupt) if target_acc < 1.0 else 0
        num_corrupt = min(num_corrupt, total)

        if num_corrupt > 0:
            # 批量选择要修改的索引（NumPy高效）
            corrupt_indices = np.random.choice(total, num_corrupt, replace=False)
            # 获取这些索引对应的原始标签
            true_labels = [ds.base_labels[ds.indices[idx]] for idx in corrupt_indices]
            # 批量生成错误标签（向量化操作）
            possible_labels = [
                [c for c in range(num_classes) if c != tl] for tl in true_labels
            ]
            # 随机选择错误标签并更新
            for idx, pos_labels in zip(corrupt_indices, possible_labels):
                if pos_labels:
                    ds.custom_labels[idx] = np.random.choice(pos_labels)

    # 8. 打印报告（按需执行，不影响核心效率）
    if print_report:
        print_distribution(original_clients, num_classes, "标签调整前分布")
        print_distribution(clients, num_classes, "标签调整后分布")
        # 计算并打印准确率
        accuracies = calculate_accuracy(original_clients, clients)
        print("\n===== 标签准确率报告 =====")
        avg_acc = 0.0
        for i, (cid, acc) in enumerate(accuracies.items()):
            print(f"客户端 {cid} (目标: {target_accuracies[i]:.2f}): 实际准确率 = {acc:.4f}")
            avg_acc += acc
        print(f"平均准确率: {avg_acc / num_client:.4f}")

    return clients, base_images, base_labels


# 验证数据加载效率
def verify_loader(clients, client_id):
    ds = clients[client_id]
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    # 测试加载速度（批量读取）
    batch_images, batch_labels = next(iter(loader))
    print(f"\n验证客户端 {client_id} 加载器:")
    print(f"  批量大小: {len(batch_labels)}, 标签示例: {batch_labels[:5].tolist()}")


if __name__ == "__main__":
    print("测试非IID分布（k=1）：")
    clients, _, _ = divide_data(
        num_client=5,
        num_local_class=2,
        target_accuracies=[0.0, 0.2, 0.5, 0.8, 1.0],
        k=1
    )
    verify_loader(clients, 'f_00000')

    print("\n\n测试非IID分布（k=2）：")
    clients, _, _ = divide_data(
        num_client=5,
        num_local_class=2,
        target_accuracies=[0.0, 0.0, 0.0, 0.0, 0.0],
        k=2
    )
    verify_loader(clients, 'f_00003')