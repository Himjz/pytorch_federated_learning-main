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
from tqdm import tqdm


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
        self.class_to_idx = {}
        self.processed_trainset = None  # 新增：存储处理后的训练集（应用了subset和cut）
        self.processed_testset = None  # 新增：存储处理后的测试集（应用了subset和cut）

        self.is_loaded = False
        self.is_divided = False

        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load(self) -> 'UniversalDataLoader':
        if self.is_loaded:
            print("数据集已加载")
            return self

        default_datasets = {
            'MNIST': (torchvision.datasets.MNIST, 10),
            'CIFAR10': (torchvision.datasets.CIFAR10, 10),
            'FashionMNIST': (torchvision.datasets.FashionMNIST, 10),
            'CIFAR100': (torchvision.datasets.CIFAR100, 100),
        }

        transform_list = [transforms.ToTensor()]

        if self.augmentation:
            transform_list = self._get_augmentation_transform() + transform_list

        auto_detect_size = self.image_size is None

        channel_transform = self._create_channel_transform()
        if channel_transform:
            transform_list.append(channel_transform)

        self._add_normalization(transform_list)

        if self.dataset_name in default_datasets:
            self._load_default_dataset(default_datasets, transform_list, auto_detect_size)
        else:
            self._load_custom_dataset(transform_list, auto_detect_size)

        # 确保使用处理后的数据集（应用了subset和cut）
        if self.processed_trainset is not None:
            self.original_trainset = self.processed_trainset
        if self.processed_testset is not None:
            self.testset = self.processed_testset

        if self.selected_classes is None:
            if hasattr(self.original_trainset, 'classes'):
                self.selected_classes = self.original_trainset.classes
                self.class_to_idx = self.original_trainset.class_to_idx
            elif self.num_classes > 0:
                self.selected_classes = list(range(self.num_classes))
                self.class_to_idx = {str(i): i for i in range(self.num_classes)}
            else:
                unique_labels = torch.unique(self.base_labels).tolist()
                self.selected_classes = unique_labels
                self.class_to_idx = {str(lbl): lbl for lbl in unique_labels}

        if self.base_labels is not None and len(self.base_labels) > 0:
            unique_labels = torch.unique(self.base_labels).tolist()
        else:
            print("警告: 未检测到任何样本标签")

        self.is_loaded = True
        print(f"数据集 {self.dataset_name} 加载完成")
        return self

    def divide(self, create: bool = False) -> Union[Tuple[Dict, Dataset], 'UniversalDataLoader']:
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

    def _fallback_client_creation(self, class_indices: Dict[int, List[int]]) -> None:
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

    def _get_augmentation_transform(self) -> List[transforms.transforms]:
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
        if self.in_channels is None:
            return None

        class ChannelTransformer:
            def __init__(self, target_channels: int):
                self.target_channels = target_channels

            def __call__(self, img: torch.Tensor) -> torch.Tensor:
                if img.shape[0] == self.target_channels:
                    return img
                if self.target_channels == 3 and img.shape[0] == 1:
                    return img.repeat(3, 1, 1)
                if self.target_channels == 1 and img.shape[0] == 3:
                    return torch.mean(img, dim=0, keepdim=True)
                return img

        return ChannelTransformer(self.in_channels)

    def _add_normalization(self, transform_list: List[transforms.transforms]) -> None:
        if self.in_channels == 1 or (self.in_channels is None and self.dataset_name in ['MNIST', 'FashionMNIST']):
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    def _load_default_dataset(self, default_datasets: Dict, base_transform_list: List, auto_detect_size: bool) -> None:
        dataset_cls, self.num_classes = default_datasets[self.dataset_name]
        os.makedirs(self.root, exist_ok=True)

        base_transform = transforms.Compose(base_transform_list)

        full_trainset = dataset_cls(root=self.root, train=True, download=self.download, transform=base_transform)
        full_testset = dataset_cls(root=self.root, train=False, download=self.download, transform=base_transform)
        self.original_trainset, self.original_testset = full_trainset, full_testset

        self.class_to_idx = {str(i): i for i in range(self.num_classes)}
        if hasattr(full_trainset, 'classes'):
            self.selected_classes = full_trainset.classes
            self.class_to_idx = full_trainset.class_to_idx

        # 应用类别筛选和裁剪，并保存到processed_trainset和processed_testset
        trainset, testset = self._apply_class_subset(full_trainset, full_testset)

        # 处理尺寸
        final_transform, trainset, testset = self._handle_size(
            dataset_cls, base_transform_list, full_trainset, full_testset,
            trainset, testset, auto_detect_size
        )

        # 应用裁剪
        if self.cut_ratio is not None:
            trainset, testset = self._cut_samples(trainset, testset, self.cut_ratio)

        # 保存处理后的数据集引用
        self.processed_trainset = trainset
        self.processed_testset = testset

        if self.in_channels is None:
            self.in_channels = self._detect_in_channels(trainset)

        if self.save:
            self._save_processed_dataset(
                dataset_cls, final_transform,
                full_trainset, full_testset,
                trainset, testset,
                self.root
            )

        self._process_labels_and_images(trainset, testset)

    def _load_custom_dataset(self, base_transform_list: List, auto_detect_size: bool) -> None:
        dataset_root = os.path.join(self.root, self.dataset_name)

        if not os.path.exists(dataset_root):
            raise ValueError(f"数据集目录不存在: {dataset_root}")

        original_train_dir = os.path.join(dataset_root, 'train')
        if not os.path.exists(original_train_dir):
            raise ValueError(f"训练集目录不存在: {original_train_dir}")

        all_classes = sorted([cls for cls in os.listdir(original_train_dir)
                              if os.path.isdir(os.path.join(original_train_dir, cls))])

        if not all_classes:
            raise ValueError(f"在 {original_train_dir} 中未找到任何类别文件夹")

        self.num_classes = len(all_classes)
        print(f"发现 {self.num_classes} 个类别")

        selected_classes = all_classes
        if self.subset_params is not None:
            subset_type, subset_value = self.subset_params
            selected_classes = self._select_custom_classes(all_classes, subset_type, subset_value)
            self.selected_classes = selected_classes

            if len(self.selected_classes) < 1:
                raise ValueError(f"选择的类别数小于1，当前数量: {len(self.selected_classes)}")
        else:
            self.selected_classes = all_classes

        self.class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

        temp_transform = transforms.Compose([transforms.ToTensor()])
        temp_dataset = torchvision.datasets.ImageFolder(root=original_train_dir, transform=temp_transform)

        if len(temp_dataset) == 0:
            raise ValueError(f"在 {original_train_dir} 中未找到任何图像文件")
        print(f"临时数据集加载成功，包含 {len(temp_dataset)} 个样本")

        transform_list = self._handle_custom_dataset_size(base_transform_list, temp_dataset, auto_detect_size)

        if self.in_channels is None:
            self.in_channels = self._detect_in_channels(temp_dataset)
            if self.in_channels not in [1, 3]:
                raise ValueError(f"不支持的通道数: {self.in_channels}，请手动指定in_channels参数")

            channel_transform = self._create_channel_transform()
            if channel_transform:
                transform_list.insert(len(transform_list) - 1, channel_transform)

        final_transform = transforms.Compose(transform_list)

        processed_dataset_name = self._get_processed_dataset_name()
        processed_dataset_root = os.path.join(self.root, processed_dataset_name)

        create_processed = self.save and not os.path.exists(processed_dataset_root)
        if create_processed:
            print(f"将创建处理后的数据集到: {processed_dataset_root}")
            self._create_processed_custom_dataset(
                dataset_root, processed_dataset_root,
                selected_classes, self.cut_ratio,
                final_transform
            )
        else:
            if not os.path.exists(processed_dataset_root):
                print(f"处理后的数据集不存在，将直接从原始数据集加载: {original_train_dir}")

        train_dir = os.path.join(processed_dataset_root, 'train') if create_processed else original_train_dir
        val_dir = os.path.join(processed_dataset_root, 'val') if create_processed else os.path.join(dataset_root, 'val')

        if not os.path.isdir(train_dir):
            raise ValueError(f"训练集目录不存在: {train_dir}")

        train_files = []
        for cls in selected_classes:
            cls_dir = os.path.join(train_dir, cls)
            if os.path.isdir(cls_dir):
                train_files.extend([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])

        if not train_files:
            raise ValueError(f"在 {train_dir} 中未找到任何图像文件")
        print(f"训练集目录包含 {len(train_files)} 个图像文件")

        # 加载原始数据集
        orig_trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=final_transform)

        # 应用裁剪（如果指定）
        if self.cut_ratio is not None:
            train_size = int(len(orig_trainset) * self.cut_ratio)
            train_indices = np.random.choice(len(orig_trainset), train_size, replace=False)
            trainset = Subset(orig_trainset, train_indices)
            print(f"已应用裁剪: 保留 {self.cut_ratio * 100:.1f}% 的样本，剩余 {len(trainset)} 个样本")
        else:
            trainset = orig_trainset

        # 保存处理后的数据集引用
        self.processed_trainset = trainset

        self.testset = None
        if os.path.isdir(val_dir):
            orig_testset = torchvision.datasets.ImageFolder(root=val_dir, transform=final_transform)

            # 对测试集应用同样的裁剪
            if self.cut_ratio is not None:
                test_size = int(len(orig_testset) * self.cut_ratio)
                test_indices = np.random.choice(len(orig_testset), test_size, replace=False)
                self.testset = Subset(orig_testset, test_indices)
                print(f"已应用裁剪: 测试集保留 {self.cut_ratio * 100:.1f}% 的样本，剩余 {len(self.testset)} 个样本")
            else:
                self.testset = orig_testset

            print(f"验证集包含 {len(self.testset)} 个样本")
        else:
            print(f"警告: 验证集目录不存在 {val_dir}")

        self.base_images = [trainset[i][0].to(self.device, non_blocking=True) for i in
                            tqdm(range(len(trainset)), desc="加载训练集", unit="样本")] if self.preload_to_gpu else [
            trainset[i][0] for i in range(len(trainset))]

        if isinstance(trainset, Subset):
            # 对于Subset，获取真实标签
            if isinstance(trainset.dataset.targets, torch.Tensor):
                self.base_labels = trainset.dataset.targets[trainset.indices].detach().clone().to(dtype=torch.int64)
            else:
                self.base_labels = torch.tensor([trainset.dataset.targets[i] for i in trainset.indices],
                                                dtype=torch.int64)
        else:
            if isinstance(trainset.targets, torch.Tensor):
                self.base_labels = trainset.targets.detach().clone().to(dtype=torch.int64)
            else:
                self.base_labels = torch.tensor(trainset.targets, dtype=torch.int64)

        self.indices = list(range(len(trainset)))
        self.num_classes = len(trainset.dataset.classes) if isinstance(trainset, Subset) else len(trainset.classes)

        print(f"成功加载自定义数据集: {train_dir}")
        print(f"训练集样本数: {len(trainset)}, 验证集样本数: {len(self.testset) if self.testset else 0}")
        print(f"图像尺寸: {self.image_size}, 通道数: {self.in_channels}")
        print(
            f"选择的类别数: {self.num_classes}, 类别列表: {trainset.dataset.classes if isinstance(trainset, Subset) else trainset.classes}")

    def _apply_class_subset(self, full_trainset: Dataset, full_testset: Dataset) -> Tuple[Dataset, Dataset]:
        if self.subset_params is not None:
            subset_type, subset_value = self.subset_params
            trainset, testset, selected_classes = self._select_classes(
                full_trainset, full_testset, subset_type, subset_value)
            self.selected_classes = selected_classes
            print(f"已应用类别筛选: 选择了 {len(selected_classes)} 个类别")
        else:
            trainset = full_trainset
            testset = full_testset
            if hasattr(full_trainset, 'classes'):
                self.selected_classes = full_trainset.classes
                self.class_to_idx = full_trainset.class_to_idx
            else:
                self.selected_classes = list(range(self.num_classes))
                self.class_to_idx = {str(i): i for i in range(self.num_classes)}

        if len(self.selected_classes) < 1:
            raise ValueError(f"选择的类别数小于1，当前数量: {len(self.selected_classes)}")

        return trainset, testset

    def _handle_size(self, dataset_cls: Any, base_transform_list: List, full_trainset: Dataset,
                     full_testset: Dataset, trainset: Dataset, testset: Dataset, auto_detect_size: bool) -> Tuple[
        Any, Dataset, Dataset]:
        final_transform = transforms.Compose(base_transform_list)

        if auto_detect_size:
            self.image_size = self._detect_image_size(trainset)
            new_transform_list = [transforms.Resize(self.image_size)] + base_transform_list
            final_transform = transforms.Compose(new_transform_list)

            full_trainset = dataset_cls(root=self.root, train=True, download=self.download, transform=final_transform)
            full_testset = dataset_cls(root=self.root, train=False, download=self.download, transform=final_transform)
            self.original_trainset, self.original_testset = full_trainset, full_testset

            # 重新应用类别筛选以确保与新的transform匹配
            trainset, testset = self._apply_class_subset(full_trainset, full_testset)
        else:
            if isinstance(self.image_size, int):
                base_transform_list.insert(0, transforms.Resize((self.image_size, self.image_size)))
            else:
                base_transform_list.insert(0, transforms.Resize(self.image_size))
            final_transform = transforms.Compose(base_transform_list)

            full_trainset = dataset_cls(root=self.root, train=True, download=self.download, transform=final_transform)
            full_testset = dataset_cls(root=self.root, train=False, download=self.download, transform=final_transform)
            self.original_trainset, self.original_testset = full_trainset, full_testset

            # 重新应用类别筛选以确保与新的transform匹配
            trainset, testset = self._apply_class_subset(full_trainset, full_testset)

        return final_transform, trainset, testset

    def _handle_custom_dataset_size(self, base_transform_list: List, temp_dataset: Dataset,
                                    auto_detect_size: bool) -> List:
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
        # 处理训练集标签
        if isinstance(trainset, Subset):
            # 对于Subset，获取正确的标签
            if hasattr(trainset.dataset, 'targets'):
                if isinstance(trainset.dataset.targets, torch.Tensor):
                    self.base_labels = trainset.dataset.targets[trainset.indices].detach().clone().to(dtype=torch.int64)
                else:
                    self.base_labels = torch.tensor([trainset.dataset.targets[i] for i in trainset.indices],
                                                    dtype=torch.int64)
            elif hasattr(trainset.dataset, 'labels'):
                if isinstance(trainset.dataset.labels, torch.Tensor):
                    self.base_labels = trainset.dataset.labels[trainset.indices].detach().clone().to(dtype=torch.int64)
                else:
                    self.base_labels = torch.tensor([trainset.dataset.labels[i] for i in trainset.indices],
                                                    dtype=torch.int64)
        else:
            if hasattr(trainset, 'targets'):
                if isinstance(trainset.targets, torch.Tensor):
                    self.base_labels = trainset.targets.detach().clone().to(dtype=torch.int64)
                else:
                    self.base_labels = torch.tensor(trainset.targets, dtype=torch.int64)
            elif hasattr(trainset, 'labels'):
                if isinstance(trainset.labels, torch.Tensor):
                    self.base_labels = trainset.labels.detach().clone().to(dtype=torch.int64)
                else:
                    self.base_labels = torch.tensor(trainset.labels, dtype=torch.int64)

        # 处理测试集标签
        if isinstance(testset, Subset):
            if hasattr(testset.dataset, 'targets'):
                if isinstance(testset.dataset.targets, torch.Tensor):
                    testset.targets = testset.dataset.targets[testset.indices].detach().clone().to(dtype=torch.int64)
                else:
                    testset.targets = torch.tensor([testset.dataset.targets[i] for i in testset.indices],
                                                   dtype=torch.int64)
            elif hasattr(testset.dataset, 'labels'):
                if isinstance(testset.dataset.labels, torch.Tensor):
                    testset.targets = testset.dataset.labels[testset.indices].detach().clone().to(dtype=torch.int64)
                else:
                    testset.targets = torch.tensor([testset.dataset.labels[i] for i in testset.indices],
                                                   dtype=torch.int64)
        else:
            if hasattr(testset, 'targets'):
                if isinstance(testset.targets, torch.Tensor):
                    testset.targets = testset.targets.detach().clone().to(dtype=torch.int64)
                else:
                    testset.targets = torch.tensor(testset.targets, dtype=torch.int64)
            elif hasattr(testset, 'labels'):
                if isinstance(testset.labels, torch.Tensor):
                    testset.targets = testset.labels.detach().clone().to(dtype=torch.int64)
                else:
                    testset.targets = torch.tensor(testset.labels, dtype=torch.int64)

        # 加载图像
        self.base_images = [trainset[i][0].to(self.device, non_blocking=True) for i in
                            tqdm(range(len(trainset)), desc="加载训练集", unit="样本")] if self.preload_to_gpu else [
            trainset[i][0] for i in range(len(trainset))]

        self.indices = list(range(len(trainset)))
        self.testset = testset

    def _group_indices_by_class(self) -> Dict[int, List[int]]:
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
        client_classes = {}
        unique_labels = list(torch.unique(self.base_labels).numpy())

        for i in range(self.num_client):
            client_class_indices = [(i * self.k + j) % len(unique_labels) for j in range(self.num_local_class)]
            client_classes[f'f_{i:05d}'] = [unique_labels[idx] for idx in client_class_indices]
        return client_classes

    def _count_class_assignments(self, client_classes: Dict[str, List[int]]) -> Dict[int, int]:
        class_assign_counts = defaultdict(int)
        for classes in client_classes.values():
            for cls in classes:
                class_assign_counts[cls] += 1
        return class_assign_counts

    def _create_clients(self, client_classes: Dict[str, List[int]], class_indices: Dict[int, List[int]],
                        class_assign_counts: Dict[int, int]) -> None:
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

    def _detect_image_size(self, dataset: Dataset) -> Tuple[int, int]:
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
        print(f"自动检测到图像尺寸: {most_common_size}")
        return most_common_size

    def _detect_in_channels(self, dataset: Dataset) -> int:
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
        if not self.is_loaded:
            raise RuntimeError("请先调用load()方法加载数据集")

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
        train_size = int(len(trainset) * cut_ratio)
        test_size = int(len(testset) * cut_ratio)

        print(f"应用样本裁剪: 保留 {cut_ratio * 100:.1f}% 的样本")
        print(f"裁剪前 - 训练集: {len(trainset)} 样本, 测试集: {len(testset)} 样本")

        train_indices = np.random.choice(len(trainset), train_size, replace=False)
        test_indices = np.random.choice(len(testset), test_size, replace=False)

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

        print(f"裁剪后 - 训练集: {len(train_subset)} 样本, 测试集: {len(test_subset)} 样本")
        return train_subset, test_subset

    def _get_processed_dataset_name(self) -> str:
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

            for i in tqdm(train_indices, desc="保存训练集", unit="样本"):
                img, label = full_trainset[i]
                label_dir = os.path.join(train_dir, str(label))
                os.makedirs(label_dir, exist_ok=True)
                img_path = os.path.join(label_dir, f"img_{i}.png")
                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)
                img.save(img_path)

            for i in tqdm(test_indices, desc="保存测试集", unit="样本"):
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
        self._process_and_save_custom_dataset(
            os.path.join(original_root, 'train'),
            os.path.join(processed_root, 'train'),
            selected_classes, cut_ratio, transform
        )

        val_dir = os.path.join(original_root, 'val')
        if os.path.exists(val_dir):
            self._process_and_save_custom_dataset(
                val_dir,
                os.path.join(processed_root, 'val'),
                selected_classes, cut_ratio, transform
            )
        else:
            print(f"警告: 原始验证集目录不存在 {val_dir}")

    def _process_and_save_custom_dataset(self, original_dir: str, processed_dir: str,
                                         selected_classes: List[str], cut_ratio: Optional[float],
                                         transform: Any) -> None:
        os.makedirs(processed_dir, exist_ok=True)
        total_saved = 0

        for class_name in selected_classes:
            class_path = os.path.join(original_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"警告: 类别目录不存在 {class_path}")
                continue

            processed_class_path = os.path.join(processed_dir, class_name)
            os.makedirs(processed_class_path, exist_ok=True)

            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(image_extensions) and
                      os.path.isfile(os.path.join(class_path, f))]

            if not images:
                print(f"警告: 在 {class_path} 中未找到任何图像文件")
                continue

            if cut_ratio is not None:
                num_to_keep = max(1, int(len(images) * cut_ratio))
                selected_images = random.sample(images, num_to_keep)
                print(f"裁剪类别 {class_name}: 保留 {cut_ratio * 100:.1f}% 的样本，从 {len(images)} 到 {num_to_keep}")
            else:
                selected_images = images

            for img in tqdm(selected_images, desc=f"处理 {class_name}", unit="图像"):
                src = os.path.join(class_path, img)
                try:
                    img_obj = Image.open(src)
                    if img_obj.mode in ['RGBA', 'P']:
                        img_obj = img_obj.convert('RGB')
                    transformed_img = transform(img_obj)
                    save_img = transforms.ToPILImage()(transformed_img)
                    dst = os.path.join(processed_class_path, img)
                    save_img.save(dst)
                    total_saved += 1
                except Exception as e:
                    print(f"处理图像 {src} 时出错: {e}")

        print(f"已处理并保存 {total_saved} 个图像文件到 {processed_dir}")

    def _print_report(self) -> None:
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

    def get_test_loader(self, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
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
        if torch.is_tensor(label):
            return label.item()
        return label

    def __len__(self) -> int:
        if not self.is_loaded:
            return 0
        return len(self.indices) if self.indices is not None else 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        if not self.is_loaded:
            raise RuntimeError("请先调用load()方法加载数据集")

        if idx >= len(self.indices):
            raise IndexError(f"索引 {idx} 超出范围，数据集大小为 {len(self.indices)}")

        base_idx = self.indices[idx]
        label = self.custom_labels.get(idx, self.base_labels[base_idx])

        if self.base_images is None:
            image, _ = self.original_trainset[base_idx]
            image = image.to(self.device, non_blocking=True)
        else:
            image = self.base_images[base_idx].to(self.device, non_blocking=True)

        return image, label


class Client(Dataset):
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

        if len(self) > 0:
            self.apply_strategy()
        else:
            print(f"警告: 客户端 {client_id} 没有样本")

    def apply_strategy(self) -> None:
        if self.strategy == 'feature_noise':
            self.noise_scale = self.strategy_ratio
            brightness_range = min(0.2, self.strategy_ratio * 0.5)
            self.brightness_factor = 1.0 + (random.random() - 0.5) * brightness_range
        elif self.strategy == 'malicious':
            num_corrupt = max(0, min(len(self) - 1, int(len(self) * (1 - self.strategy_ratio))))

            if num_corrupt > 0 and len(self) > 0:
                try:
                    indices = torch.randperm(len(self))[:num_corrupt].tolist()

                    for idx in indices:
                        data_idx = self.indices[idx]
                        true_label = UniversalDataLoader._to_numpy_label(self.dataset.base_labels[data_idx])

                        if true_label in self.assigned_classes:
                            possible_labels = [c for c in self.assigned_classes if c != true_label]
                            if possible_labels:
                                self.custom_labels[idx] = np.random.choice(possible_labels)
                except Exception as e:
                    print(f"客户端 {self.client_id} 应用恶意策略时出错: {e}")

    def get_data_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=(self.device.type == 'cpu'),
            num_workers=0 if self.device.type == 'cuda' else 2
        )

    def calculate_label_accuracy(self) -> float:
        orig_labels = self.dataset.base_labels[self.indices].detach().clone().to(self.device, non_blocking=True)

        adj_labels = []
        for idx in range(len(self)):
            adj_labels.append(self.custom_labels.get(idx, self.dataset.base_labels[self.indices[idx]]))
        adj_labels = torch.tensor(adj_labels, device=self.device, dtype=torch.int64)

        return (orig_labels == adj_labels).sum().item() / len(orig_labels) if len(orig_labels) > 0 else 0.0

    def print_distribution(self, num_classes: int) -> None:
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
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 示例：使用cut和subset参数
    loader = UniversalDataLoader(
        root="../data",
        num_client=12,
        num_local_class=4,
        dataset_name='dt3',
        seed=0,
        untrusted_strategies=[2.1, 3.5, 4.3, 0.5, 1.0],
        device=device,
        preload_to_gpu=False,
        in_channels=1,
        size=32,
        #augmentation=True,
        cut=0.6,  # 保留80%的样本
        #subset=("tail",2)
    )

    loader.load()

    mean, std = loader.calculate_statistics()
    print(f"\n数据集统计信息: 均值={mean}, 标准差={std}")

    try:
        trainset_config, testset = loader.divide()

        print("\n数据集信息:")
        print(f"  类别数: {loader.num_classes}")
        print(f"  图像尺寸: {loader.image_size}x{loader.image_size}")
        print(f"  图像通道数: {loader.in_channels}")

        for i in range(3):
            client_id = f"f_0000{i}"
            if client_id in trainset_config['user_data']:
                trainset_config['user_data'][client_id].verify_loader()
            else:
                print(f"客户端 {client_id} 不存在")
    except Exception as e:
        print(f"划分数据集时出错: {e}")
