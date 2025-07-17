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
        self.custom_labels = custom_labels if custom_labels is not None else {}  # 仅存储修改的标签

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        label = self.custom_labels.get(idx, self.base_labels[base_idx])  # 优先使用修改后的标签
        return self.base_images[base_idx], label


def load_base_data(root='dt2'):
    """加载原始数据（仅加载一次，返回引用）"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # 加载训练集和测试集
    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(root, 'train'),
        transform=transform
    )
    testset = torchvision.datasets.ImageFolder(
        root=os.path.join(root, 'val'),
        transform=transform
    )

    # 预加载图像和标签（仅一次）
    base_images = [trainset[i][0] for i in range(len(trainset))]
    base_labels = [trainset.targets[i] for i in range(len(trainset))]
    num_classes = len(trainset.classes)

    return base_images, base_labels, num_classes, testset


def print_distribution(clients, num_classes, title):
    """高效打印分布：批量统计标签"""
    print(f"\n===== {title} =====")
    for client_id, ds in clients.items():
        labels = [ds.custom_labels.get(idx, ds.base_labels[ds.indices[idx]]) for idx in range(len(ds))]
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
        orig_labels = [orig_ds.base_labels[orig_ds.indices[idx]] for idx in range(len(orig_ds))]
        adj_labels = [adj_ds.custom_labels.get(idx, adj_ds.base_labels[adj_ds.indices[idx]]) for idx in
                      range(len(adj_ds))]
        correct = np.sum(np.array(orig_labels) == np.array(adj_labels))
        accuracies[client_id] = correct / len(orig_labels) if orig_labels else 0.0
    return accuracies


def divide_data(
        num_client=5,
        num_local_class=2,
        target_accuracies=[0.0, 0.2, 0.5, 0.8, 1.0],
        k=1,
        print_report=True
):
    # 1. 加载原始数据和测试集（新增测试集返回）
    base_images, base_labels, num_classes, testset = load_base_data()
    np.random.seed(0)

    # 2. 按类别预分组索引
    class_indices = {cls: [] for cls in range(num_classes)}
    for idx, cls in enumerate(base_labels):
        class_indices[cls].append(idx)
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

    # 5. 分配样本索引
    client_indices = {cid: [] for cid in client_classes}
    for client_id, classes in client_classes.items():
        for cls in classes:
            total = len(class_indices[cls])
            if class_assign_counts[cls] == 0:
                continue
            per_client = total // class_assign_counts[cls] if class_assign_counts[cls] > 1 else total
            client_indices[client_id].extend(class_indices[cls][:per_client])
            class_indices[cls] = class_indices[cls][per_client:]
            class_assign_counts[cls] -= 1

    # 6. 创建客户端数据集
    clients = {}
    original_clients = {}
    for client_id in client_classes:
        indices = client_indices[client_id]
        original_clients[client_id] = EfficientDataset(base_images, base_labels, indices)
        clients[client_id] = EfficientDataset(base_images, base_labels, indices)

    # 7. 批量修改标签
    for i, (client_id, ds) in enumerate(clients.items()):
        target_acc = target_accuracies[i]
        total = len(ds)
        if total == 0 or target_acc >= 1.0:
            continue

        num_corrupt = max(1, int(total * (1 - target_acc)))
        num_corrupt = min(num_corrupt, total)
        corrupt_indices = np.random.choice(total, num_corrupt, replace=False)
        true_labels = [ds.base_labels[ds.indices[idx]] for idx in corrupt_indices]
        possible_labels = [[c for c in range(num_classes) if c != tl] for tl in true_labels]

        for idx, pos_labels in zip(corrupt_indices, possible_labels):
            if pos_labels:
                ds.custom_labels[idx] = np.random.choice(pos_labels)

    # 8. 封装训练集配置（保持与旧接口一致）
    trainset_config = {
        'users': list(clients.keys()),
        'user_data': clients,
        'num_samples': sum(len(ds) for ds in clients.values())
    }

    # 9. 打印报告
    if print_report:
        print_distribution(original_clients, num_classes, "标签调整前分布")
        print_distribution(clients, num_classes, "标签调整后分布")
        accuracies = calculate_accuracy(original_clients, clients)
        print("\n===== 标签准确率报告 =====")
        avg_acc = sum(accuracies.values()) / len(accuracies) if accuracies else 0.0
        for i, (cid, acc) in enumerate(accuracies.items()):
            print(f"客户端 {cid} (目标: {target_accuracies[i]:.2f}): 实际准确率 = {acc:.4f}")
        print(f"平均准确率: {avg_acc:.4f}")

    # 仅返回2个值：训练集配置和测试集（与旧接口兼容）
    return trainset_config, testset


# 验证数据加载器
def verify_loader(client_data, client_id):
    ds = client_data[client_id]
    loader = DataLoader(ds, batch_size=5, shuffle=False)
    batch_imgs, batch_labels = next(iter(loader))
    print(f"\n验证客户端 {client_id} 的加载器:")
    print(f"  返回标签: {batch_labels.tolist()}")
    print(f"  预期标签: {[ds.custom_labels.get(idx, ds.base_labels[ds.indices[idx]]) for idx in range(5)]}")
    print("  ✅ 验证通过" if batch_labels.tolist() == [ds.custom_labels.get(idx, ds.base_labels[ds.indices[idx]]) for idx
                                                      in range(5)] else "  ❌ 验证失败")


if __name__ == "__main__":
    # 测试接口兼容性
    trainset_config, testset = divide_data(print_report=True)
    verify_loader(trainset_config['user_data'], 'f_00000')