import os

import numpy as np
import torch
from PIL import Image


def to_numpy_label(label):
    """将标签转换为普通整数"""
    return label.item() if torch.is_tensor(label) else label


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
        labels = [int(label) if torch.is_tensor(label) else int(label) for label in labels]
        counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
        print(f"客户端 {client_id}:")
        for cls in range(num_classes):
            if counts[cls] > 0:
                print(f"  类别 {cls}: {counts[cls]} 个样本")
        print(f"  特征噪声样本数: {ds.noisy_mask.sum().item()}" if ds.noisy_mask.any() else "")
        print(f"  数据稀疏: {getattr(ds, 'is_data_sparse', False)}")
        print(f"  类别倾斜: {getattr(ds, 'is_class_skewed', False)}")
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


def detect_dominant_image_size(data_dir):
    """检测数据集中占比最高的图像尺寸"""
    size_counts = {}
    sample_count = 0
    max_samples = 1000  # 最多检查1000张图像以确定尺寸

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        size = (width, height)
                        size_counts[size] = size_counts.get(size, 0) + 1
                        sample_count += 1
                        if sample_count >= max_samples:
                            break
                except Exception as e:
                    print(f"无法读取图像 {img_path}: {e}")
        if sample_count >= max_samples:
            break

    if not size_counts:
        raise ValueError(f"无法从 {data_dir} 中检测到图像尺寸")

    # 找出最常见的尺寸
    dominant_size = max(size_counts, key=size_counts.get)
    if dominant_size[0] != dominant_size[1]:
        print(f"警告: 检测到非正方形图像尺寸 {dominant_size}，将使用较长边作为统一尺寸")
        return max(dominant_size)

    return dominant_size[0]