import random
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Any

import numpy as np
import torch
from tqdm import tqdm

from .client import Client
from .dataset_loader import DatasetLoader
from .clients_controller import ClientsController


class DataSplitter(DatasetLoader):
    """数据集划分器，负责将数据集划分给多个客户端"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.export = kwargs.get('export', False)
        self.clients = ClientsController(self.class_to_idx, self.processed_testset,
                                         self.export, self.original_trainset, self.root,
                                         self.dataset_name, self.in_channels, self.base_images)
        self.is_divided = False

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
                    'users': list(self.clients.get_clients().keys()),
                    'user_data': self.clients,
                    'num_samples': sum(len(client) for client in self.clients.get_clients().values())
                }
                return trainset_config, self.testset

        class_indices = self._group_indices_by_class()

        valid_classes = [cls for cls, indices in class_indices.items() if len(indices) > 0]
        if not valid_classes:
            print("尝试使用标签值作为类别重新分组...")
            alternative_class_indices = defaultdict(list)
            for idx in range(len(self.base_labels)):
                lbl = self.to_numpy_label(self.base_labels[idx])
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

        # 根据分布类型分配客户端类别
        client_classes = self._assign_classes_to_clients(valid_classes)

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

        print(f"数据集 {self.dataset_name} 已划分为 {len(self.clients.get_clients())} 个客户端 (请求的数量: {self.num_client})")
        self.clients.export()
        if create:
            return self
        else:
            return {
                'users': list(self.clients.get_clients().keys()),
                'user_data': self.clients.get_clients(),
                'num_samples': sum(len(client) for client in self.clients.get_clients().values())
            }, self.testset

    def _group_indices_by_class(self) -> Dict[int, List[int]]:
        """按类别分组样本索引"""
        if self.selected_classes is None:
            unique_labels = torch.unique(self.base_labels).tolist()
            self.selected_classes = unique_labels

        class_indices = defaultdict(list)
        for idx in tqdm(self.indices, desc="按类别分组", unit="样本"):
            cls = self.to_numpy_label(self.base_labels[idx])
            class_indices[cls].append(idx)

        for indices in class_indices.values():
            np.random.shuffle(indices)

        print("\n类别样本分布:")
        for cls, indices in class_indices.items():
            print(f"标签 {cls}: {len(indices)} 个样本")

        return class_indices

    def _assign_classes_to_clients(self, valid_classes: List[int]) -> Dict[str, List[int]]:
        """
        为每个客户端分配类别
        根据distribution_type使用不同的分配策略
        """
        client_classes = {}
        num_classes = len(valid_classes)
        class_indices = {cls: idx for idx, cls in enumerate(valid_classes)}

        if self.distribution_type == "default":
            # 使用原有的轮询策略，参数来自distribution_param
            k = self.distribution_param if self.distribution_param is not None else 1
            for i in range(self.num_client):
                client_class_indices = [(i * k + j) % num_classes for j in range(self.num_local_class)]
                client_classes[f'f_{i:05d}'] = [valid_classes[idx] for idx in client_class_indices]

        elif self.distribution_type == "dirichlet":
            # 使用Dirichlet分布实现Non-IID分配
            alpha = self.distribution_param if self.distribution_param is not None else 0.5
            print(f"使用Dirichlet分布分配类别，浓度参数alpha={alpha}")

            # 为每个客户端生成类别分布
            dirichlet_samples = np.random.dirichlet(
                alpha=[alpha] * num_classes,
                size=self.num_client
            )

            for i in range(self.num_client):
                # 按概率排序，取概率最高的num_local_class个类别
                class_probs = dirichlet_samples[i]
                top_classes_idx = np.argsort(class_probs)[-self.num_local_class:][::-1]
                client_classes[f'f_{i:05d}'] = [valid_classes[idx] for idx in top_classes_idx]

        else:
            raise ValueError(f"不支持的分布类型: {self.distribution_type}，请使用'default'或'dirichlet'")

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
                self.clients.append_client(Client(
                    client_id=client_id,
                    dataset=self,
                    indices=client_indices,
                    assigned_classes=classes,
                    strategy=strategy,
                    strategy_ratio=ratio,
                    device=self.device
                ))

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
                cls = self.to_numpy_label(self.base_labels[idx])
                client_labels.add(cls)

            strategy = 'normal'
            ratio = 1.0
            if self.untrusted_strategies and i < len(self.untrusted_strategies):
                strategy, ratio = self.parse_strategy(self.untrusted_strategies[i])

            self.clients.append_client(Client(
                client_id=f'f_{i:05d}',
                dataset=self,
                indices=client_indices,
                assigned_classes=list(client_labels),
                strategy=strategy,
                strategy_ratio=ratio,
                device=self.device
            ))

    def _print_report(self) -> None:
        """打印详细的数据集划分报告"""
        # 收集所有客户端的样本数量
        client_sample_counts = [len(client) for client in self.clients.get_clients().values()]
        total_samples = sum(client_sample_counts)
        avg_samples = np.mean(client_sample_counts)
        std_samples = np.std(client_sample_counts)
        min_samples = np.min(client_sample_counts)
        max_samples = np.max(client_sample_counts)

        # 计算类别分布统计
        class_distribution = defaultdict(int)
        client_class_coverage = []

        for client in self.clients.get_clients().values():
            # 统计每个客户端的类别分布
            client_classes = defaultdict(int)
            for idx in client.train_indices:
                cls = self.to_numpy_label(self.base_labels[idx])
                class_distribution[cls] += 1
                client_classes[cls] += 1

            # 计算每个客户端的类别覆盖率
            client_class_count = len(client_classes)
            client_class_coverage.append(client_class_count / len(self.selected_classes) * 100)

        avg_class_coverage = np.mean(client_class_coverage)

        # 打印报告
        print("\n" + "=" * 60)
        print(f"          联邦数据集划分详细报告 (设备: {self.device})          ")
        print("=" * 60)

        # 基本信息部分
        print("\n[基本信息]")
        print(f"  数据集名称:         {self.dataset_name}")
        print(f"  图像尺寸:           {self.image_size}, 通道数: {self.in_channels}")
        print(f"  分布类型:           {self.distribution_type}")
        print(f"  分布参数:           {self.distribution_param}")
        print(f"  客户端数量:         {len(self.clients.get_clients())} (请求数量: {self.num_client})")
        print(f"  总样本数:           {total_samples}")
        print(f"  测试集样本数:       {len(self.testset) if self.testset else 0}")

        # 样本分布统计部分
        print("\n[样本分布统计]")
        print(f"  客户端平均样本数:   {avg_samples:.2f} ± {std_samples:.2f}")
        print(f"  最小样本数客户端:   {min_samples}")
        print(f"  最大样本数客户端:   {max_samples}")
        print(f"  样本分布均衡性:     {1 - (std_samples / avg_samples) if avg_samples > 0 else 0:.4f}")

        # 类别分布部分
        print("\n[类别分布]")
        print(f"  总类别数:           {len(self.selected_classes)}")
        print(f"  客户端平均类别覆盖率: {avg_class_coverage:.2f}%")

        print("\n  类别样本分布:")
        total = sum(class_distribution.values())
        for cls in sorted(class_distribution.keys()):
            count = class_distribution[cls]
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"    类别 {cls}: {count} 个样本 ({percentage:.2f}%)")

        # 客户端策略分布部分
        strategy_counts = defaultdict(int)
        for client in self.clients.get_clients().values():
            strategy_counts[client.strategy] += 1

        print("\n[客户端策略分布]")
        for strategy, count in strategy_counts.items():
            percentage = (count / len(self.clients.get_clients())) * 100
            print(f"  {strategy}: {count} 个客户端 ({percentage:.2f}%)")

        # 恶意客户端报告部分
        malicious_clients = [c for c in self.clients.get_clients().values() if c.strategy == 'malicious']
        if malicious_clients:
            print("\n[恶意客户端标签准确率]")
            total_accuracy = 0.0
            for client in malicious_clients:
                acc = client.calculate_label_accuracy()
                print(f"  客户端 {client.client_id}: {acc:.4f}")
                total_accuracy += acc

            avg_accuracy = total_accuracy / len(malicious_clients)
            print(f"  平均准确率: {avg_accuracy:.4f}")

        # 高级选项部分
        print("\n[高级选项]")
        if self.subset_params is not None:
            subset_type, subset_value = self.subset_params
            print(f"  选择类别:           类型={subset_type}, 数量={subset_value}")
            print(f"  选中的类别:         {self.selected_classes}")

        if self.cut_ratio is not None:
            print(f"  裁剪样本比例:       {self.cut_ratio}")

        print(f"  数据增强:           {'已启用' if self.augmentation else '未启用'}")

        print("\n" + "=" * 60)
        print("                报告结束                ")
        print("=" * 60 + "\n")

    def load(self) -> 'DatasetLoader':
        res = super().load()
        self.clients = ClientsController(self.class_to_idx, self.processed_testset,
                                         self.export, self.original_trainset, self.root,
                                         self.dataset_name, self.in_channels, self.base_images)
        return res
