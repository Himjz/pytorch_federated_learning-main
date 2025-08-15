import random
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Any

import numpy as np
import torch
from tqdm import tqdm

from .client import Client
from .dataset_loader import DatasetLoader


class DataSplitter(DatasetLoader):
    """数据集划分器，负责将数据集划分给多个客户端"""
    
    def __init__(self, **kwargs):
        super().__init__(** kwargs)
        self.clients = {}  # 存储客户端
        self.is_divided = False  # 标记是否已划分
    
    def divide(self, create: bool = False) -> Union[Tuple[Dict, Any], 'DataSplitter']:
        """
        将数据集划分给多个客户端
        
        Args:
            create: 是否返回自身而非划分结果
            
        Returns:
            如果create为True，返回自身；否则返回划分后的训练集配置和测试集
        """
        if not self.is_loaded:
            raise RuntimeError("请先调用load()方法加载数据集")

        if self.base_labels is None or len(self.base_labels) == 0:
            raise RuntimeError("数据集中没有样本，请检查数据集加载是否正确")

        if not self.class_to_idx:
            raise RuntimeError("类别到索引的映射为空，请检查数据集")

        if self.selected_classes is None or len(self.selected_classes) == 0:
            unique_labels = torch.unique(self.base_labels).tolist()
            self.selected_classes = unique_labels
            if len(self.selected_classes) == 0:
                raise ValueError("无法确定数据集的类别，请检查数据加载是否正确")

        if self.is_divided:
            print("数据集已划分")
            if create:
                return self
            else:
                trainset_config = {
                    'users': list(self.clients.keys()),
                    'user_data': self.clients,
                    'num_samples': sum(len(client) for client in self.clients.values())
                }
                return trainset_config, self.testset

        class_indices = self._group_indices_by_class()

        valid_classes = [cls for cls, indices in class_indices.items() if len(indices) > 0]
        if not valid_classes:
            print("尝试使用标签值作为类别重新分组...")
            alternative_class_indices = defaultdict(list)
            for idx in range(len(self.base_labels)):
                lbl = self._to_numpy_label(self.base_labels[idx])
                alternative_class_indices[lbl].append(idx)

            valid_classes = [cls for cls, indices in alternative_class_indices.items() if len(indices) > 0]
            if valid_classes:
                class_indices = alternative_class_indices
                self.selected_classes = valid_classes
                print(f"使用标签值作为类别: {valid_classes}")
            else:
                raise RuntimeError("没有可用的样本分配给客户端，请检查数据集")

        adjusted_num_local_class = min(self.num_local_class, len(valid_classes))
        if adjusted_num_local_class != self.num_local_class:
            print(f"警告: 可用类别数少于请求的每个客户端类别数，已调整为 {adjusted_num_local_class}")
            self.num_local_class = adjusted_num_local_class

        client_classes = self._assign_classes_to_clients()

        class_assign_counts = self._count_class_assignments(client_classes)

        self._create_clients(client_classes, class_indices, class_assign_counts)

        if not self.clients:
            print("尝试使用备用策略分配样本...")
            self._fallback_client_creation(class_indices)

            if not self.clients:
                raise RuntimeError("未能创建任何客户端，请检查数据分配逻辑")

        self.is_divided = True

        if self.print_report:
            self._print_report()

        print(f"数据集 {self.dataset_name} 已划分为 {len(self.clients)} 个客户端 (请求的数量: {self.num_client})")

        if create:
            return self
        else:
            return {
                'users': list(self.clients.keys()),
                'user_data': self.clients,
                'num_samples': sum(len(client) for client in self.clients.values())
            }, self.testset
    
    def _group_indices_by_class(self) -> Dict[int, List[int]]:
        """按类别分组样本索引"""
        if self.selected_classes is None:
            unique_labels = torch.unique(self.base_labels).tolist()
            self.selected_classes = unique_labels

        class_indices = defaultdict(list)
        for idx in tqdm(self.indices, desc="按类别分组", unit="样本"):
            cls = self._to_numpy_label(self.base_labels[idx])
            class_indices[cls].append(idx)

        for indices in class_indices.values():
            np.random.shuffle(indices)

        print("\n类别样本分布:")
        for cls, indices in class_indices.items():
            print(f"标签 {cls}: {len(indices)} 个样本")

        return class_indices
    
    def _assign_classes_to_clients(self) -> Dict[str, List[int]]:
        """为每个客户端分配类别"""
        client_classes = {}
        unique_labels = list(torch.unique(self.base_labels).numpy())

        for i in range(self.num_client):
            client_class_indices = [(i * self.k + j) % len(unique_labels) for j in range(self.num_local_class)]
            client_classes[f'f_{i:05d}'] = [unique_labels[idx] for idx in client_class_indices]
        return client_classes
    
    def _count_class_assignments(self, client_classes: Dict[str, List[int]]) -> Dict[int, int]:
        """统计每个类别被分配给多少个客户端"""
        class_assign_counts = defaultdict(int)
        for classes in client_classes.values():
            for cls in classes:
                class_assign_counts[cls] += 1
        return class_assign_counts
    
    def _create_clients(self, client_classes: Dict[str, List[int]], class_indices: Dict[int, List[int]],
                        class_assign_counts: Dict[int, int]) -> None:
        """创建客户端并分配数据"""
        self.clients = {}
        remaining_indices = {cls: indices.copy() for cls, indices in class_indices.items()}

        for client_id, classes in tqdm(client_classes.items(), desc="创建客户端", unit="客户端"):
            client_index = int(client_id.split('_')[-1])

            strategy_code = self.untrusted_strategies[client_index] if self.untrusted_strategies and client_index < len(
                self.untrusted_strategies) else 1.0
            strategy, ratio = self.parse_strategy(strategy_code)

            dominant_cls = random.choice(classes) if strategy == 'class_skew' else None

            client_indices = []
            for cls in classes:
                if class_assign_counts[cls] <= 0 or len(remaining_indices.get(cls, [])) == 0:
                    continue

                total_samples = len(remaining_indices[cls])
                clients_remaining = class_assign_counts[cls]
                per_client = max(1, total_samples // clients_remaining)

                if strategy == 'sparse':
                    per_client = max(1, int(per_client * ratio))
                elif strategy == 'class_skew':
                    per_client = per_client * 2 if cls == dominant_cls else max(1, int(per_client * (1 - ratio)))

                per_client = min(per_client, len(remaining_indices[cls]))

                client_indices.extend(remaining_indices[cls][:per_client])
                remaining_indices[cls] = remaining_indices[cls][per_client:]
                class_assign_counts[cls] -= 1

            if len(client_indices) == 0 and len(classes) > 0:
                for cls in classes:
                    if len(remaining_indices.get(cls, [])) > 0:
                        client_indices.append(remaining_indices[cls].pop(0))
                        class_assign_counts[cls] -= 1
                        break
                    elif len(class_indices.get(cls, [])) > 0:
                        client_indices.append(class_indices[cls].pop())
                        break

            if len(client_indices) > 0:
                self.clients[client_id] = Client(
                    client_id=client_id,
                    dataset=self,
                    indices=client_indices,
                    assigned_classes=classes,
                    strategy=strategy,
                    strategy_ratio=ratio,
                    device=self.device
                )
    
    def _fallback_client_creation(self, class_indices: Dict[int, List[int]]) -> None:
        """备用客户端创建策略"""
        all_indices = []
        for indices in class_indices.values():
            all_indices.extend(indices)

        if not all_indices:
            return

        random.shuffle(all_indices)

        samples_per_client = max(1, len(all_indices) // self.num_client)

        for i in range(self.num_client):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < self.num_client - 1 else len(all_indices)
            client_indices = all_indices[start_idx:end_idx]

            if not client_indices:
                continue

            client_labels = set()
            for idx in client_indices:
                cls = self._to_numpy_label(self.base_labels[idx])
                client_labels.add(cls)

            strategy = 'normal'
            ratio = 1.0
            if self.untrusted_strategies and i < len(self.untrusted_strategies):
                strategy, ratio = self.parse_strategy(self.untrusted_strategies[i])

            self.clients[f'f_{i:05d}'] = Client(
                client_id=f'f_{i:05d}',
                dataset=self,
                indices=client_indices,
                assigned_classes=list(client_labels),
                strategy=strategy,
                strategy_ratio=ratio,
                device=self.device
            )
    
    def _print_report(self) -> None:
        """打印数据集划分报告"""
        print(f"\n===== 联邦数据集报告（设备: {self.device}） =====")
        print(f"数据集名称: {self.dataset_name}")
        print(f"图像尺寸: {self.image_size}, 通道数: {self.in_channels}")

        if self.subset_params is not None:
            subset_type, subset_value = self.subset_params
            print(f"选择类别: 类型={subset_type}, 数量={subset_value}")
            print(f"选中的类别: {self.selected_classes}")
            print(f"选中的类别数量: {len(self.selected_classes)}")

        if self.cut_ratio is not None:
            print(f"裁剪样本比例: {self.cut_ratio}")

        if self.augmentation:
            print(f"数据增强: 已启用")

        print(f"总样本数: {len(self)}")
        print(f"测试集样本数: {len(self.testset) if self.testset else 0}")
        print(f"客户端数量: {len(self.clients)}")

        malicious_clients = [c for c in self.clients.values() if c.strategy == 'malicious']
        if malicious_clients:
            print("\n===== 标签准确率报告 =====")
            total_accuracy = 0.0
            for client in malicious_clients:
                acc = client.calculate_label_accuracy()
                print(f"客户端 {client.client_id}: 准确率 = {acc:.4f}")
                total_accuracy += acc

            avg_accuracy = total_accuracy / len(malicious_clients)
            print(f"恶意客户端平均准确率: {avg_accuracy:.4f}")
