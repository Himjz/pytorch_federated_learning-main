import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset


class UniversalDataLoader(Dataset):
    def __init__(self,
                 dataset_name: str = 'MNIST',
                 num_client: int = 1,
                 num_local_class: int = 10,
                 untrusted_strategies: Optional[List[float]] = None,
                 k: int = 1,
                 print_report: bool = True,
                 device: Optional[torch.device] = None,
                 preload_to_gpu: bool = False,
                 seed: int = 0,
                 root: str = './data',
                 download: bool = True,
                 cut: Optional[float] = None,
                 save: bool = False,
                 subset: Optional[Tuple[str, Union[int, float]]] = None,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 in_channels: Optional[int] = None,
                 augmentation: bool = False,
                 augmentation_params: Optional[Dict[str, Any]] = None):
        # 初始化参数
        self.dataset_name = dataset_name
        self.num_client = num_client
        self.num_local_class = num_local_class
        self.untrusted_strategies = untrusted_strategies
        self.k = k
        self.print_report = print_report
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preload_to_gpu = preload_to_gpu
        self.seed = seed
        self.root = root
        self.download = download
        self.cut_ratio = cut
        self.save = save
        self.subset_params = subset
        self.image_size = size
        self.in_channels = in_channels
        self.augmentation = augmentation
        self.augmentation_params = augmentation_params or {
            'rotation': 15,
            'translation': 0.1,
            'scale': 0.1,
            'flip': True
        }

        # 初始化数据相关变量
        self.base_images = None
        self.base_labels = None
        self.indices = None
        self.custom_labels = {}
        self.clients = {}
        self.testset = None
        self.num_classes = 0
        self.original_trainset = None
        self.original_testset = None
        self.selected_classes = None

        # 状态标记
        self.is_loaded = False
        self.is_divided = False

        # 设置随机种子
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """统一设置随机种子，确保实验可重复性"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load(self) -> 'UniversalDataLoader':
        """加载数据集"""
        if self.is_loaded:
            print("数据集已加载，无需重复加载")
            return self

        # 定义支持的默认数据集
        default_datasets = {
            'MNIST': (torchvision.datasets.MNIST, 10),
            'CIFAR10': (torchvision.datasets.CIFAR10, 10),
            'FashionMNIST': (torchvision.datasets.FashionMNIST, 10),
            'CIFAR100': (torchvision.datasets.CIFAR100, 100),
        }

        # 创建基础转换列表
        transform_list = [transforms.ToTensor()]

        # 添加数据增强
        if self.augmentation:
            transform_list = self._get_augmentation_transform() + transform_list

        # 处理尺寸
        auto_detect_size = self.image_size is None

        # 处理通道数转换
        channel_transform = self._create_channel_transform()
        if channel_transform:
            transform_list.append(channel_transform)

        # 添加归一化
        self._add_normalization(transform_list)

        # 加载数据集
        if self.dataset_name in default_datasets:
            self._load_default_dataset(default_datasets, transform_list, auto_detect_size)
        else:
            self._load_custom_dataset(transform_list, auto_detect_size)

        self.is_loaded = True
        print(f"数据集 {self.dataset_name} 加载完成")
        return self

    def divide(self, create: bool = False) -> Union[Tuple[Dict, Dataset], 'UniversalDataLoader']:
        """划分联邦数据集，创建客户端"""
        if not self.is_loaded:
            raise RuntimeError("请先调用load()方法加载数据集")

        if self.is_divided:
            print("数据集已划分，无需重复划分")
            if create:
                return self
            else:
                # 构建符合要求的训练集配置
                trainset_config = {
                    'users': list(self.clients.keys()),
                    'user_data': self.clients,
                    'num_samples': sum(len(client) for client in self.clients.values())
                }
                return trainset_config, self.testset

        # 按类别分组索引
        class_indices = self._group_indices_by_class()

        # 为每个客户端分配类别
        client_classes = self._assign_classes_to_clients()

        # 计算每个类别需要分配给多少客户端
        class_assign_counts = self._count_class_assignments(client_classes)

        # 创建客户端
        self._create_clients(client_classes, class_indices, class_assign_counts)

        self.is_divided = True

        if self.print_report:
            self._print_report()

        print(f"数据集 {self.dataset_name} 已划分为 {self.num_client} 个客户端")

        # 根据create参数返回不同结果
        if create:
            return self
        else:
            return {
                'users': list(self.clients.keys()),
                'user_data': self.clients,
                'num_samples': sum(len(client) for client in self.clients.values())
            }, self.testset

    def _get_augmentation_transform(self) -> List[transforms.transforms]:
        """获取数据增强转换"""
        aug_transforms = []
        if self.augmentation_params.get('rotation', 0) > 0:
            aug_transforms.append(transforms.RandomRotation(self.augmentation_params['rotation']))
        if self.augmentation_params.get('flip', False):
            aug_transforms.append(transforms.RandomHorizontalFlip())
        if self.augmentation_params.get('translation', 0) > 0:
            translate = self.augmentation_params['translation']
            aug_transforms.append(transforms.RandomAffine(0, translate=(translate, translate)))
        if self.augmentation_params.get('scale', 0) > 0:
            scale = self.augmentation_params['scale']
            aug_transforms.append(transforms.RandomAffine(0, scale=(1 - scale, 1 + scale)))
        return aug_transforms

    def _create_channel_transform(self) -> Optional[Any]:
        """创建通道转换"""
        if self.in_channels is None:
            return None

        class ChannelTransformer:
            def __init__(self, target_channels: int):
                self.target_channels = target_channels

            def __call__(self, img: torch.Tensor) -> torch.Tensor:
                if img.shape[0] == self.target_channels:
                    return img
                # 灰度转RGB
                if self.target_channels == 3 and img.shape[0] == 1:
                    return img.repeat(3, 1, 1)
                # RGB转灰度
                if self.target_channels == 1 and img.shape[0] == 3:
                    return torch.mean(img, dim=0, keepdim=True)
                return img

        return ChannelTransformer(self.in_channels)

    def _add_normalization(self, transform_list: List[transforms.transforms]) -> None:
        """添加归一化转换"""
        if self.in_channels == 1 or (self.in_channels is None and self.dataset_name in ['MNIST', 'FashionMNIST']):
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    def _load_default_dataset(self, default_datasets: Dict, base_transform_list: List, auto_detect_size: bool) -> None:
        """加载默认数据集（如MNIST, CIFAR等）"""
        dataset_cls, self.num_classes = default_datasets[self.dataset_name]
        os.makedirs(self.root, exist_ok=True)

        # 初始转换（不含尺寸调整）
        base_transform = transforms.Compose(base_transform_list)

        # 加载完整数据集
        full_trainset = dataset_cls(root=self.root, train=True, download=self.download, transform=base_transform)
        full_testset = dataset_cls(root=self.root, train=False, download=self.download, transform=base_transform)
        self.original_trainset, self.original_testset = full_trainset, full_testset

        # 应用类别筛选
        trainset, testset = self._apply_class_subset(full_trainset, full_testset)

        # 处理尺寸
        final_transform, trainset, testset = self._handle_size(
            dataset_cls, base_transform_list, full_trainset, full_testset,
            trainset, testset, auto_detect_size
        )

        # 自动检测通道数（如果未指定）
        if self.in_channels is None:
            self.in_channels = self._detect_in_channels(trainset)

        # 裁剪样本
        if self.cut_ratio is not None:
            trainset, testset = self._cut_samples(trainset, testset, self.cut_ratio)

        # 保存处理后的数据集
        if self.save:
            self._save_processed_dataset(
                dataset_cls, final_transform,
                full_trainset, full_testset,
                trainset, testset,
                self.root
            )

        # 处理标签并加载数据
        self._process_labels_and_images(trainset, testset)

    def _load_custom_dataset(self, base_transform_list: List, auto_detect_size: bool) -> None:
        """加载自定义数据集"""
        dataset_root = os.path.join(self.root, self.dataset_name)

        if not os.path.exists(dataset_root):
            raise ValueError(f"数据集目录不存在: {dataset_root}")

        original_train_dir = os.path.join(dataset_root, 'train')
        all_classes = sorted([cls for cls in os.listdir(original_train_dir)
                              if os.path.isdir(os.path.join(original_train_dir, cls))])
        self.num_classes = len(all_classes)

        # 应用类别筛选
        selected_classes = all_classes
        if self.subset_params is not None:
            subset_type, subset_value = self.subset_params
            selected_classes = self._select_custom_classes(all_classes, subset_type, subset_value)
            self.selected_classes = selected_classes

            if len(self.selected_classes) < 1:
                raise ValueError(f"选择的类别数小于1，当前数量: {len(self.selected_classes)}")

        # 处理尺寸和转换
        temp_transform = transforms.Compose([transforms.ToTensor()])
        temp_dataset = torchvision.datasets.ImageFolder(root=original_train_dir, transform=temp_transform)

        # 处理尺寸
        transform_list = self._handle_custom_dataset_size(base_transform_list, temp_dataset, auto_detect_size)

        # 自动检测通道数（如果未指定）
        if self.in_channels is None:
            self.in_channels = self._detect_in_channels(temp_dataset)
            if self.in_channels not in [1, 3]:
                raise ValueError(f"不支持的通道数: {self.in_channels}，请手动指定in_channels参数")

            # 添加通道转换
            channel_transform = self._create_channel_transform()
            if channel_transform:
                transform_list.insert(len(transform_list) - 1, channel_transform)

        final_transform = transforms.Compose(transform_list)

        # 处理保存
        processed_dataset_name = self._get_processed_dataset_name()
        processed_dataset_root = os.path.join(self.root, processed_dataset_name)

        if self.save and not os.path.exists(processed_dataset_root):
            self._create_processed_custom_dataset(
                dataset_root, processed_dataset_root,
                selected_classes, self.cut_ratio,
                final_transform
            )

        # 加载最终数据集
        train_dir = os.path.join(processed_dataset_root, 'train')
        val_dir = os.path.join(processed_dataset_root, 'val')

        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise ValueError(f"数据集目录结构不符合要求，需要包含'train'和'val'子目录: {processed_dataset_root}")

        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=final_transform)
        self.testset = torchvision.datasets.ImageFolder(root=val_dir, transform=final_transform)

        # 处理标签和图像 - 修复张量转换警告
        self.base_images = [trainset[i][0].to(self.device, non_blocking=True) for i in
                            range(len(trainset))] if self.preload_to_gpu else [
            trainset[i][0] for i in range(len(trainset))]

        # 修复自定义数据集标签转换警告
        if isinstance(trainset.targets, torch.Tensor):
            self.base_labels = trainset.targets.detach().clone().to(dtype=torch.int64)
        else:
            self.base_labels = torch.tensor(trainset.targets, dtype=torch.int64)

        self.indices = list(range(len(trainset)))
        self.num_classes = len(trainset.classes)

        print(f"成功加载自定义数据集: {processed_dataset_root}")
        print(f"训练集样本数: {len(trainset)}, 验证集样本数: {len(self.testset)}")
        print(f"图像尺寸: {self.image_size}, 通道数: {self.in_channels}")
        print(f"选择的类别数: {self.num_classes}, 类别列表: {trainset.classes}")

    def _apply_class_subset(self, full_trainset: Dataset, full_testset: Dataset) -> Tuple[Dataset, Dataset]:
        """应用类别筛选"""
        if self.subset_params is not None:
            subset_type, subset_value = self.subset_params
            trainset, testset, selected_classes = self._select_classes(
                full_trainset, full_testset, subset_type, subset_value)
            self.selected_classes = selected_classes
        else:
            trainset = full_trainset
            testset = full_testset
            self.selected_classes = list(range(self.num_classes))

        if len(self.selected_classes) < 1:
            raise ValueError(f"选择的类别数小于1，当前数量: {len(self.selected_classes)}")

        return trainset, testset

    def _handle_size(self, dataset_cls: Any, base_transform_list: List, full_trainset: Dataset,
                     full_testset: Dataset, trainset: Dataset, testset: Dataset, auto_detect_size: bool) -> Tuple[
        Any, Dataset, Dataset]:
        """处理图像尺寸"""
        final_transform = transforms.Compose(base_transform_list)

        if auto_detect_size:
            # 自动检测尺寸
            self.image_size = self._detect_image_size(trainset)
            # 添加尺寸调整
            new_transform_list = [transforms.Resize(self.image_size)] + base_transform_list
            final_transform = transforms.Compose(new_transform_list)

            # 重新加载数据集以应用新尺寸
            full_trainset = dataset_cls(root=self.root, train=True, download=self.download, transform=final_transform)
            full_testset = dataset_cls(root=self.root, train=False, download=self.download, transform=final_transform)
            self.original_trainset, self.original_testset = full_trainset, full_testset

            # 重新应用类别筛选
            trainset, testset = self._apply_class_subset(full_trainset, full_testset)
        else:
            # 使用指定尺寸
            if isinstance(self.image_size, int):
                base_transform_list.insert(0, transforms.Resize((self.image_size, self.image_size)))
            else:
                base_transform_list.insert(0, transforms.Resize(self.image_size))
            final_transform = transforms.Compose(base_transform_list)

            # 重新加载数据集以应用尺寸
            full_trainset = dataset_cls(root=self.root, train=True, download=self.download, transform=final_transform)
            full_testset = dataset_cls(root=self.root, train=False, download=self.download, transform=final_transform)
            self.original_trainset, self.original_testset = full_trainset, full_testset

            # 重新应用类别筛选
            trainset, testset = self._apply_class_subset(full_trainset, full_testset)

        return final_transform, trainset, testset

    def _handle_custom_dataset_size(self, base_transform_list: List, temp_dataset: Dataset,
                                    auto_detect_size: bool) -> List:
        """处理自定义数据集的尺寸"""
        transform_list = base_transform_list.copy()

        if auto_detect_size:
            self.image_size = self._detect_image_size(temp_dataset)
            transform_list.insert(0, transforms.Resize(self.image_size))
        else:
            if isinstance(self.image_size, int):
                transform_list.insert(0, transforms.Resize((self.image_size, self.image_size)))
            else:
                transform_list.insert(0, transforms.Resize(self.image_size))

        return transform_list

    def _process_labels_and_images(self, trainset: Dataset, testset: Dataset) -> None:
        """处理标签和图像数据 - 修复所有张量转换警告"""
        # 处理标签
        if hasattr(trainset, 'targets'):
            # 修复警告：使用detach().clone()代替直接转换
            if isinstance(trainset.targets, torch.Tensor):
                trainset.targets = trainset.targets.detach().clone().to(dtype=torch.int64)
            else:
                trainset.targets = torch.tensor(trainset.targets, dtype=torch.int64)

            if isinstance(testset.targets, torch.Tensor):
                testset.targets = testset.targets.detach().clone().to(dtype=torch.int64)
            else:
                testset.targets = torch.tensor(testset.targets, dtype=torch.int64)
        elif hasattr(trainset, 'labels'):
            # 修复警告：使用detach().clone()代替直接转换
            if isinstance(trainset.labels, torch.Tensor):
                trainset.targets = trainset.labels.detach().clone().to(dtype=torch.int64)
            else:
                trainset.targets = torch.tensor(trainset.labels, dtype=torch.int64)

            if isinstance(testset.labels, torch.Tensor):
                testset.targets = testset.labels.detach().clone().to(dtype=torch.int64)
            else:
                testset.targets = torch.tensor(testset.labels, dtype=torch.int64)
        else:
            orig_train_targets = self.original_trainset.targets if hasattr(self.original_trainset,
                                                                           'targets') else self.original_trainset.labels
            orig_test_targets = self.original_testset.targets if hasattr(self.original_testset,
                                                                         'targets') else self.original_testset.labels

            # 修复警告：使用detach().clone()代替直接转换
            if isinstance(orig_train_targets, torch.Tensor):
                trainset.targets = orig_train_targets[trainset.indices].detach().clone().to(dtype=torch.int64)
            else:
                trainset.targets = torch.tensor([orig_train_targets[i] for i in trainset.indices], dtype=torch.int64)

            if isinstance(orig_test_targets, torch.Tensor):
                testset.targets = orig_test_targets[testset.indices].detach().clone().to(dtype=torch.int64)
            else:
                testset.targets = torch.tensor([orig_test_targets[i] for i in testset.indices], dtype=torch.int64)

        # 加载图像和标签 - 使用non_blocking=True加速GPU传输
        self.base_images = [trainset[i][0].to(self.device, non_blocking=True) for i in
                            range(len(trainset))] if self.preload_to_gpu else [
            trainset[i][0] for i in range(len(trainset))]
        self.base_labels = trainset.targets
        self.indices = list(range(len(trainset)))
        self.testset = testset

    def _group_indices_by_class(self) -> Dict[int, List[int]]:
        """按类别分组索引"""
        class_indices = {cls: [] for cls in self.selected_classes}
        for idx in self.indices:
            cls = self._to_numpy_label(self.base_labels[idx])
            if cls in class_indices:
                class_indices[cls].append(idx)

        # 打乱每个类别的索引
        for indices in class_indices.values():
            np.random.shuffle(indices)

        return class_indices

    def _assign_classes_to_clients(self) -> Dict[str, List[int]]:
        """为每个客户端分配类别"""
        client_classes = {}
        for i in range(self.num_client):
            client_class_indices = [(i * self.k + j) % len(self.selected_classes) for j in range(self.num_local_class)]
            client_classes[f'f_{i:05d}'] = [self.selected_classes[idx] for idx in client_class_indices]
        return client_classes

    def _count_class_assignments(self, client_classes: Dict[str, List[int]]) -> Dict[int, int]:
        """计算每个类别需要分配给多少客户端"""
        class_assign_counts = {cls: 0 for cls in self.selected_classes}
        for classes in client_classes.values():
            for cls in classes:
                class_assign_counts[cls] += 1
        return class_assign_counts

    def _create_clients(self, client_classes: Dict[str, List[int]], class_indices: Dict[int, List[int]],
                        class_assign_counts: Dict[int, int]) -> None:
        """创建客户端，优化样本分配算法"""
        self.clients = {}
        # 先复制一份类别索引，避免修改原始数据
        remaining_indices = {cls: indices.copy() for cls, indices in class_indices.items()}

        for client_id, classes in client_classes.items():
            client_index = int(client_id.split('_')[-1])

            # 解析策略
            strategy_code = self.untrusted_strategies[client_index] if self.untrusted_strategies and client_index < len(
                self.untrusted_strategies) else 1.0
            strategy, ratio = self.parse_strategy(strategy_code)

            dominant_cls = random.choice(classes) if strategy == 'class_skew' else None

            # 收集客户端数据索引
            client_indices = []
            for cls in classes:
                if class_assign_counts[cls] <= 0 or len(remaining_indices[cls]) == 0:
                    continue

                # 计算每个客户端应分配的样本数
                total_samples = len(remaining_indices[cls])
                clients_remaining = class_assign_counts[cls]
                per_client = max(1, total_samples // clients_remaining)

                # 应用策略调整
                if strategy == 'sparse':
                    per_client = max(1, int(per_client * ratio))
                elif strategy == 'class_skew':
                    per_client = per_client * 2 if cls == dominant_cls else max(1, int(per_client * (1 - ratio)))

                per_client = min(per_client, len(remaining_indices[cls]))

                # 分配样本并更新
                client_indices.extend(remaining_indices[cls][:per_client])
                remaining_indices[cls] = remaining_indices[cls][per_client:]
                class_assign_counts[cls] -= 1

            # 创建客户端实例
            self.clients[client_id] = Client(
                client_id=client_id,
                dataset=self,
                indices=client_indices,
                assigned_classes=classes,
                strategy=strategy,
                strategy_ratio=ratio,
                device=self.device
            )

    def _detect_image_size(self, dataset: Dataset) -> Tuple[int, int]:
        """通过取样法自动检测图像尺寸"""
        sample_size = min(int(len(dataset) * 0.2), 1000)
        if sample_size < 1:
            sample_size = 1

        size_counts = defaultdict(int)
        indices = random.sample(range(len(dataset)), sample_size)

        for i in indices:
            img, _ = dataset[i]
            if isinstance(img, torch.Tensor):
                size = (img.shape[1], img.shape[2])
            elif isinstance(img, Image.Image):
                size = img.size
            else:
                continue

            size_counts[size] += 1

        if not size_counts:
            raise ValueError("无法检测图像尺寸，请手动指定size参数")

        most_common_size = max(size_counts.items(), key=lambda x: x[1])[0]
        print(f"自动检测到图像尺寸: {most_common_size} (基于{sample_size}个样本)")
        return most_common_size

    def _detect_in_channels(self, dataset: Dataset) -> int:
        """自动检测图像通道数"""
        sample_size = min(5, len(dataset))
        channels = set()

        for i in range(sample_size):
            img, _ = dataset[i]
            if isinstance(img, torch.Tensor):
                channels.add(img.shape[0])
            elif isinstance(img, Image.Image):
                channels.add(len(img.getbands()))

        if len(channels) != 1:
            raise ValueError(f"检测到多种通道数: {channels}，请手动指定in_channels参数")

        in_channels = channels.pop()
        print(f"自动检测到通道数: {in_channels}")
        return in_channels

    def calculate_statistics(self, sample_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算数据集的均值和标准差"""
        if not self.is_loaded:
            raise RuntimeError("请先调用load()方法加载数据集")

        # 随机采样计算统计量
        indices = random.sample(range(len(self)), min(sample_size, len(self)))

        if self.base_images is not None:
            images = [self.base_images[i] for i in indices]
        else:
            images = [self.original_trainset[i][0] for i in indices]

        images = torch.stack(images)

        mean = images.mean(dim=(0, 2, 3))
        std = images.std(dim=(0, 2, 3))

        return mean, std

    def _select_classes(self, trainset: Dataset, testset: Dataset, subset_type: str,
                        subset_value: Union[int, float]) -> Tuple[Subset, Subset, List[int]]:
        """选择指定类别的样本"""
        all_classes = list(range(self.num_classes))

        if isinstance(subset_value, float):
            num_classes = max(1, min(int(self.num_classes * subset_value), self.num_classes))
        else:
            num_classes = max(1, min(subset_value, self.num_classes))

        if subset_type == 'top':
            selected_classes = all_classes[:num_classes]
        elif subset_type == 'tail':
            selected_classes = all_classes[-num_classes:] if num_classes > 0 else []
        else:
            selected_classes = random.sample(all_classes, num_classes)

        # 筛选训练集和测试集
        train_indices = []
        test_indices = []

        train_labels = trainset.targets if hasattr(trainset, 'targets') else trainset.labels
        test_labels = testset.targets if hasattr(testset, 'targets') else testset.labels

        for i, label in enumerate(train_labels):
            if self._to_numpy_label(label) in selected_classes:
                train_indices.append(i)

        for i, label in enumerate(test_labels):
            if self._to_numpy_label(label) in selected_classes:
                test_indices.append(i)

        return Subset(trainset, train_indices), Subset(testset, test_indices), selected_classes

    def _select_custom_classes(self, all_classes: List[str], subset_type: str,
                               subset_value: Union[int, float]) -> List[str]:
        """选择自定义数据集的类别"""
        if isinstance(subset_value, float):
            num_classes = max(1, min(int(len(all_classes) * subset_value), len(all_classes)))
        else:
            num_classes = max(1, min(subset_value, len(all_classes)))

        if subset_type == 'top':
            return all_classes[:num_classes]
        elif subset_type == 'tail':
            return all_classes[-num_classes:] if num_classes > 0 else []
        else:
            return random.sample(all_classes, num_classes)

    def _cut_samples(self, trainset: Dataset, testset: Dataset, cut_ratio: float) -> Tuple[Subset, Subset]:
        """裁剪样本"""
        train_size = int(len(trainset) * cut_ratio)
        test_size = int(len(testset) * cut_ratio)

        train_indices = np.random.choice(len(trainset), train_size, replace=False)
        test_indices = np.random.choice(len(testset), test_size, replace=False)

        # 处理子集情况
        if isinstance(trainset, Subset):
            orig_indices = np.array(trainset.indices)
            train_subset = Subset(self.original_trainset, orig_indices[train_indices])
        else:
            train_subset = Subset(trainset, train_indices)

        if isinstance(testset, Subset):
            orig_indices = np.array(testset.indices)
            test_subset = Subset(self.original_testset, orig_indices[test_indices])
        else:
            test_subset = Subset(testset, test_indices)

        return train_subset, test_subset

    def _get_processed_dataset_name(self) -> str:
        """生成处理后的数据集名称"""
        if self.save:
            parts = [self.dataset_name]

            if self.subset_params is not None:
                subset_type, subset_value = self.subset_params
                parts.append(f"subset_{subset_type}_{subset_value}")

            if self.cut_ratio is not None:
                parts.append(f"cut_{self.cut_ratio}")

            if self.image_size is not None:
                if isinstance(self.image_size, tuple):
                    parts.append(f"size_{self.image_size[0]}x{self.image_size[1]}")
                else:
                    parts.append(f"size_{self.image_size}")

            if self.in_channels is not None:
                parts.append(f"channels_{self.in_channels}")

            if self.augmentation:
                parts.append("augmented")

            return "_".join(parts)
        return self.dataset_name

    def _save_processed_dataset(self, dataset_cls: Any, transform: Any, full_trainset: Dataset,
                                full_testset: Dataset, trainset: Dataset, testset: Dataset, root: str) -> None:
        """保存处理后的数据集"""
        processed_name = self._get_processed_dataset_name()
        processed_root = os.path.join(root, processed_name)
        os.makedirs(processed_root, exist_ok=True)

        if hasattr(full_trainset, 'data_path') or hasattr(full_trainset, 'images'):
            train_dir = os.path.join(processed_root, 'train')
            test_dir = os.path.join(processed_root, 'test')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            train_indices = self._get_original_indices(trainset)
            test_indices = self._get_original_indices(testset)

            # 保存训练集
            for i in train_indices:
                img, label = full_trainset[i]
                label_dir = os.path.join(train_dir, str(label))
                os.makedirs(label_dir, exist_ok=True)
                img_path = os.path.join(label_dir, f"img_{i}.png")
                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)
                img.save(img_path)

            # 保存测试集
            for i in test_indices:
                img, label = full_testset[i]
                label_dir = os.path.join(test_dir, str(label))
                os.makedirs(label_dir, exist_ok=True)
                img_path = os.path.join(label_dir, f"img_{i}.png")
                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)
                img.save(img_path)

            print(f"已保存处理后的数据集到: {processed_root}")
        else:
            print(f"警告: 数据集 {self.dataset_name} 不支持直接保存处理后的文件")

    def _get_original_indices(self, dataset: Dataset) -> List[int]:
        """获取原始数据集的索引"""
        if not isinstance(dataset, Subset):
            return list(range(len(dataset)))

        indices = dataset.indices
        while isinstance(dataset.dataset, Subset):
            dataset = dataset.dataset
            indices = [dataset.indices[i] for i in indices]

        return indices

    def _create_processed_custom_dataset(self, original_root: str, processed_root: str,
                                         selected_classes: List[str], cut_ratio: Optional[float],
                                         transform: Any) -> None:
        """创建处理后的自定义数据集"""
        # 处理训练集
        self._process_and_save_custom_dataset(
            os.path.join(original_root, 'train'),
            os.path.join(processed_root, 'train'),
            selected_classes, cut_ratio, transform
        )

        # 处理验证集
        self._process_and_save_custom_dataset(
            os.path.join(original_root, 'val'),
            os.path.join(processed_root, 'val'),
            selected_classes, cut_ratio, transform
        )

        print(f"已保存处理后的自定义数据集到: {processed_root}")

    def _process_and_save_custom_dataset(self, original_dir: str, processed_dir: str,
                                         selected_classes: List[str], cut_ratio: Optional[float],
                                         transform: Any) -> None:
        """处理并保存自定义数据集的一个子集（训练集或验证集）"""
        os.makedirs(processed_dir, exist_ok=True)

        for class_name in selected_classes:
            class_path = os.path.join(original_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            processed_class_path = os.path.join(processed_dir, class_name)
            os.makedirs(processed_class_path, exist_ok=True)

            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            if not images:
                continue

            # 裁剪样本
            if cut_ratio is not None:
                num_to_keep = max(1, int(len(images) * cut_ratio))
                selected_images = random.sample(images, num_to_keep)
            else:
                selected_images = images

            # 处理并保存图像
            for img in selected_images:
                src = os.path.join(class_path, img)
                try:
                    img_obj = Image.open(src)
                    if img_obj.mode in ['RGBA', 'P']:
                        img_obj = img_obj.convert('RGB')
                    transformed_img = transform(img_obj)
                    save_img = transforms.ToPILImage()(transformed_img)
                    dst = os.path.join(processed_class_path, img)
                    save_img.save(dst)
                except Exception as e:
                    print(f"处理图像 {src} 时出错: {e}")

    def _print_report(self) -> None:
        """打印数据集报告"""
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
            print(f"数据增强: 已启用, 参数={self.augmentation_params}")

        print(f"总样本数: {len(self)}")
        print(f"测试集样本数: {len(self.testset)}")
        print(f"客户端数量: {len(self.clients)}")

        # 打印客户端策略分配
        if self.clients and any(client.strategy != 'normal' for client in self.clients.values()):
            print("\n===== 客户端策略分配 =====")
            for client_id, client in self.clients.items():
                if client.strategy != 'normal':
                    print(f"客户端 {client_id}: 策略 = {client.strategy}, 参数 = {client.strategy_ratio:.2f}")

        # 打印客户端数据分布
        print("\n===== 客户端数据分布 =====")
        for client in self.clients.values():
            client.print_distribution(len(self.selected_classes))

        # 打印恶意客户端标签准确率
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

    def get_test_loader(self, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        """获取测试集数据加载器"""
        if not self.is_loaded:
            raise RuntimeError("请先调用load()方法加载数据集")

        if not self.testset:
            raise ValueError("没有可用的测试集")

        return DataLoader(
            self.testset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=(self.device.type == 'cpu'),
            num_workers=0 if self.device.type == 'cuda' else 2
        )

    @staticmethod
    def parse_strategy(strategy_code: Union[float, str]) -> Tuple[str, float]:
        """解析客户端策略代码"""
        try:
            code = float(strategy_code)
            integer_part = int(code)
            decimal_part = code - integer_part

            strategies = {
                0: ('malicious', decimal_part),
                2: ('sparse', decimal_part),
                3: ('feature_noise', decimal_part),
                4: ('class_skew', decimal_part),
            }

            return strategies.get(integer_part, ('normal', 1.0))
        except (ValueError, TypeError):
            return 'normal', 1.0

    @staticmethod
    def _to_numpy_label(label: Union[torch.Tensor, int]) -> int:
        """将标签转换为numpy格式"""
        if torch.is_tensor(label):
            return label.item()
        return label

    def __len__(self) -> int:
        """返回数据集大小"""
        if not self.is_loaded:
            return 0
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        """获取数据集中的一个样本"""
        if not self.is_loaded:
            raise RuntimeError("请先调用load()方法加载数据集")

        base_idx = self.indices[idx]
        label = self.custom_labels.get(idx, self.base_labels[base_idx])

        # 按需加载图像（如果未预加载）
        if self.base_images is None:
            image, _ = self.original_trainset[base_idx]
            image = image.to(self.device, non_blocking=True)
        else:
            image = self.base_images[base_idx].to(self.device, non_blocking=True)

        return image, label


class Client(Dataset):
    """客户端类，代表联邦学习中的一个客户端"""

    def __init__(self, client_id: str, dataset: UniversalDataLoader, indices: List[int],
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

        self.apply_strategy()

    def apply_strategy(self) -> None:
        """应用客户端策略"""
        if self.strategy == 'sparse':
            pass  # 稀疏策略在数据分配时已处理
        elif self.strategy == 'feature_noise':
            # 特征噪声策略
            self.noise_scale = self.strategy_ratio
            brightness_range = min(0.2, self.strategy_ratio * 0.5)
            self.brightness_factor = 1.0 + (random.random() - 0.5) * brightness_range
        elif self.strategy == 'class_skew':
            pass  # 类别倾斜策略在数据分配时已处理
        elif self.strategy == 'malicious':
            # 恶意客户端策略（标签翻转）
            num_corrupt = max(1, int(len(self) * (1 - self.strategy_ratio)))

            for idx in np.random.choice(len(self), num_corrupt, replace=False):
                data_idx = self.indices[idx]
                true_label = UniversalDataLoader._to_numpy_label(self.dataset.base_labels[data_idx])

                if true_label in self.assigned_classes:
                    possible_labels = [c for c in self.assigned_classes if c != true_label]
                    if possible_labels:
                        self.custom_labels[idx] = np.random.choice(possible_labels)

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
        """计算标签准确率（用于恶意客户端）- 修复张量转换警告"""
        # 使用张量索引代替循环，提高效率并修复警告
        orig_labels = self.dataset.base_labels[self.indices].detach().clone().to(self.device, non_blocking=True)

        adj_labels = []
        for idx in range(len(self)):
            adj_labels.append(self.custom_labels.get(idx, self.dataset.base_labels[self.indices[idx]]))
        adj_labels = torch.tensor(adj_labels, device=self.device, dtype=torch.int64)

        return (orig_labels == adj_labels).sum().item() / len(orig_labels) if len(orig_labels) > 0 else 0.0

    def print_distribution(self, num_classes: int) -> None:
        """打印客户端数据分布"""
        labels = []
        for idx in range(len(self)):
            label = self.custom_labels.get(idx, self.dataset.base_labels[self.indices[idx]])
            if torch.is_tensor(label):
                labels.append(int(label.item()))
            else:
                labels.append(int(label))

        max_class = max(self.dataset.selected_classes) if self.dataset.selected_classes else 0
        counts = torch.bincount(torch.tensor(labels, dtype=torch.int64), minlength=max_class + 1)

        print(f"客户端 {self.client_id}:")
        print(f"  总样本数: {len(self)}")
        for cls in self.dataset.selected_classes:
            if counts[cls] > 0:
                print(f"  类别 {cls}: {counts[cls]} 个样本 ({counts[cls] / len(self) * 100:.1f}%)")

        print(f"  策略: {self.strategy}, 参数: {self.strategy_ratio:.2f}")
        if self.strategy == 'feature_noise':
            print(f"  特征噪声强度: {self.noise_scale:.4f}")
            print(f"  亮度调整因子: {self.brightness_factor:.4f}")
        print()

    def verify_loader(self) -> None:
        """验证客户端数据加载器"""
        batch_size = min(5, len(self))
        if batch_size == 0:
            print(f"\n验证客户端 {self.client_id} 的加载器:")
            print("  ❌ 数据集为空，无法获取批次数据")
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
        print(f"  预期标签: {[self.custom_labels.get(idx, self.dataset.base_labels[self.indices[idx]])
                              for idx in range(batch_size)]}")

        if self.strategy == 'feature_noise':
            print(f"  噪声强度: {self.noise_scale:.4f}")

        if batch_labels.tolist() == [self.custom_labels.get(idx, self.dataset.base_labels[self.indices[idx]])
                                     for idx in range(batch_size)]:
            print("  ✅ 验证通过")
        else:
            print("  ❌ 验证失败")

    def __len__(self) -> int:
        """返回客户端数据集大小"""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        """获取客户端数据集中的一个样本"""
        data_idx = self.indices[idx]
        label = self.custom_labels.get(idx, self.dataset.base_labels[data_idx])

        # 按需加载图像（如果未预加载）
        if self.dataset.base_images is None:
            image, _ = self.dataset.original_trainset[data_idx]
            image = image.to(self.device, non_blocking=True)
        else:
            image = self.dataset.base_images[data_idx].to(self.device, non_blocking=True)

        # 应用特征噪声（如果需要）
        if self.strategy == 'feature_noise' and self.noise_scale is not None:
            noise = torch.randn_like(image, device=self.device) * self.noise_scale
            image = image + noise

            if self.brightness_factor is not None:
                image = image * self.brightness_factor

            image = torch.clamp(image, 0.0, 1.0)

        return image, label


# 使用示例
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    loader = UniversalDataLoader(
        root="../data",
        num_client=5,
        num_local_class=2,
        dataset_name='MNIST',
        seed=0,
        untrusted_strategies=[2.1, 3.5, 4.3, 0.5, 1.0],
        device=device,
        preload_to_gpu=False,
        in_channels=1,
        size=32,  # 指定图像尺寸
        augmentation=True  # 启用数据增强
    )

    loader.load()

    # 计算数据集统计信息
    mean, std = loader.calculate_statistics()
    print(f"\n数据集统计信息: 均值={mean}, 标准差={std}")

    # 划分数据集
    trainset_config, testset = loader.divide()

    print("\n数据集信息:")
    print(f"  类别数: {loader.num_classes}")
    print(f"  图像尺寸: {loader.image_size[0]}x{loader.image_size[1]}")
    print(f"  图像通道数: {loader.in_channels}")

    # 验证前3个客户端的加载器
    for i in range(3):
        trainset_config['user_data'][f"f_0000{i}"].verify_loader()
