import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Union, Tuple


class Client(Dataset):
    """客户端类，代表联邦学习中的一个客户端节点"""
    
    def __init__(self, client_id: str, dataset: Dataset, indices: List[int],
                 assigned_classes: List[int], strategy: str = 'normal', strategy_ratio: float = 1.0,
                 device: Optional[torch.device] = None):
        self.client_id = client_id
        self.dataset = dataset
        self.indices = indices
        self.assigned_classes = assigned_classes
        self.strategy = strategy
        self.strategy_ratio = strategy_ratio
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.custom_labels = {}
        self.noise_scale = None
        self.brightness_factor = None

        if len(self) > 0:
            self.apply_strategy()
        else:
            print(f"警告: 客户端 {client_id} 没有样本")
    
    def apply_strategy(self) -> None:
        """应用客户端策略（如恶意行为、数据稀疏等）"""
        if self.strategy == 'feature_noise':
            self.noise_scale = self.strategy_ratio
            brightness_range = min(0.2, self.strategy_ratio * 0.5)
            self.brightness_factor = 1.0 + (np.random.random() - 0.5) * brightness_range
        elif self.strategy == 'malicious':
            num_corrupt = max(0, min(len(self) - 1, int(len(self) * (1 - self.strategy_ratio))))

            if num_corrupt > 0 and len(self) > 0:
                try:
                    indices = torch.randperm(len(self))[:num_corrupt].tolist()

                    for idx in indices:
                        data_idx = self.indices[idx]
                        true_label = self.dataset._to_numpy_label(self.dataset.base_labels[data_idx])

                        if true_label in self.assigned_classes:
                            possible_labels = [c for c in self.assigned_classes if c != true_label]
                            if possible_labels:
                                self.custom_labels[idx] = np.random.choice(possible_labels)
                except Exception as e:
                    print(f"客户端 {self.client_id} 应用恶意策略时出错: {e}")
    
    def get_data_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """获取客户端数据加载器"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=(self.device.type == 'cpu'),
            num_workers=0 if self.device.type == 'cuda' else 2
        )
    
    def calculate_label_accuracy(self) -> float:
        """计算标签准确率（用于评估恶意客户端的标签污染程度）"""
        orig_labels = self.dataset.base_labels[self.indices].detach().clone().to(self.device, non_blocking=True)

        adj_labels = []
        for idx in range(len(self)):
            adj_labels.append(self.custom_labels.get(idx, self.dataset.base_labels[self.indices[idx]]))
        adj_labels = torch.tensor(adj_labels, device=self.device, dtype=torch.int64)

        return (orig_labels == adj_labels).sum().item() / len(orig_labels) if len(orig_labels) > 0 else 0.0
    
    def print_distribution(self, num_classes: int) -> None:
        """打印客户端数据分布信息"""
        labels = []
        for idx in range(len(self)):
            label = self.custom_labels.get(idx, self.dataset.base_labels[self.indices[idx]])
            if torch.is_tensor(label):
                labels.append(int(label.item()))
            else:
                labels.append(int(label))

        max_class = max(labels) if labels else 0
        counts = torch.bincount(torch.tensor(labels, dtype=torch.int64), minlength=max_class + 1)

        print(f"客户端 {self.client_id}:")
        print(f"  总样本数: {len(self)}")
        for cls in set(labels):
            print(f"  标签 {cls}: {counts[cls]} 个样本 ({counts[cls] / len(self) * 100:.1f}%)")

        print(f"  策略: {self.strategy}, 参数: {self.strategy_ratio:.2f}")
        print()
    
    def verify_loader(self) -> None:
        """验证客户端数据加载器是否正常工作"""
        batch_size = min(5, len(self))
        if batch_size == 0:
            print(f"\n验证客户端 {self.client_id} 的加载器:")
            print("  ❌ 数据集为空")
            return

        loader = self.get_data_loader(batch_size=batch_size, shuffle=False)

        try:
            batch_imgs, batch_labels = next(iter(loader))
        except StopIteration:
            print(f"\n验证客户端 {self.client_id} 的加载器:")
            print("  ❌ 无法获取批次数据")
            return

        print(f"\n验证客户端 {self.client_id} 的加载器:")
        print(f"  图像设备: {batch_imgs.device}")
        print(f"  图像形状: {batch_imgs.shape}")
        print(f"  返回标签: {batch_labels.tolist()}")

        if batch_labels.tolist() == [self.custom_labels.get(idx, self.dataset.base_labels[self.indices[idx]])
                                     for idx in range(batch_size)]:
            print("  ✅ 验证通过")
        else:
            print("  ❌ 验证失败")
    
    def __len__(self) -> int:
        """返回客户端样本数量"""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        """获取客户端的单个样本"""
        data_idx = self.indices[idx]
        label = self.custom_labels.get(idx, self.dataset.base_labels[data_idx])

        if self.dataset.base_images is None:
            image, _ = self.dataset.original_trainset[data_idx]
            image = image.to(self.device, non_blocking=True)
        else:
            image = self.dataset.base_images[data_idx].to(self.device, non_blocking=True)

        if self.strategy == 'feature_noise' and self.noise_scale is not None:
            noise = torch.randn_like(image, device=self.device) * self.noise_scale
            image = image + noise

            if self.brightness_factor is not None:
                image = image * self.brightness_factor

            image = torch.clamp(image, 0.0, 1.0)

        return image, label
