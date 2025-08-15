import os
import random
from typing import List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from preprocessing.utils.base_loader import BaseDataLoader
from preprocessing.utils.transforms import TransformMixin


class DatasetLoader(BaseDataLoader, TransformMixin):
    """数据集加载器，继承基础加载器和转换混合类，处理具体的数据集加载逻辑"""
    
    def __init__(self, **kwargs):
        # 初始化父类
        BaseDataLoader.__init__(self,** kwargs)
        TransformMixin.__init__(self, 
                               augmentation=kwargs.get('augmentation', False),
                               augmentation_params=kwargs.get('augmentation_params', None))
        
        # 数据集特定参数
        self.default_datasets = {
            'MNIST': (torchvision.datasets.MNIST, 10),
            'CIFAR10': (torchvision.datasets.CIFAR10, 10),
            'FashionMNIST': (torchvision.datasets.FashionMNIST, 10),
            'CIFAR100': (torchvision.datasets.CIFAR100, 100),
        }
    
    def load(self) -> 'DatasetLoader':
        """加载数据集主方法"""
        if self.is_loaded:
            print("数据集已加载")
            return self

        transform_list = [transforms.ToTensor()]

        if self.augmentation:
            transform_list = self._get_augmentation_transform() + transform_list

        auto_detect_size = self.image_size is None

        channel_transform = self._create_channel_transform()
        if channel_transform:
            transform_list.append(channel_transform)

        self._add_normalization(transform_list)

        if self.dataset_name in self.default_datasets:
            self._load_default_dataset(transform_list, auto_detect_size)
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
    
    def _load_default_dataset(self, base_transform_list: List, auto_detect_size: bool) -> None:
        """加载默认支持的数据集（如MNIST, CIFAR等）"""
        dataset_cls, self.num_classes = self.default_datasets[self.dataset_name]
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
        """加载自定义数据集"""
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
        """应用类别筛选，只保留选定的类别"""
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
        """处理图像尺寸，自动检测或调整为指定尺寸"""
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
        """处理自定义数据集的图像尺寸"""
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
        """处理标签和图像数据"""
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
    
    def _detect_image_size(self, dataset: Dataset) -> Tuple[int, int]:
        """自动检测图像尺寸"""
        sample_size = min(int(len(dataset) * 0.2), 1000)
        if sample_size < 1:
            sample_size = 1

        size_counts = {}
        indices = random.sample(range(len(dataset)), sample_size)

        for i in indices:
            img, _ = dataset[i]
            if isinstance(img, torch.Tensor):
                size = (img.shape[1], img.shape[2])
            elif isinstance(img, Image.Image):
                size = img.size
            else:
                continue

            if size in size_counts:
                size_counts[size] += 1
            else:
                size_counts[size] = 1

        if not size_counts:
            raise ValueError("无法检测图像尺寸，请手动指定size参数")

        most_common_size = max(size_counts.items(), key=lambda x: x[1])[0]
        print(f"自动检测到图像尺寸: {most_common_size}")
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
    
    def _select_classes(self, trainset: Dataset, testset: Dataset, subset_type: str,
                        subset_value: Union[int, float]) -> Tuple[Subset, Subset, List[int]]:
        """选择数据集中的部分类别"""
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
        """选择自定义数据集中的部分类别"""
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
        """按比例裁剪样本"""
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
        """获取处理后的数据集名称"""
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
        """处理并保存自定义数据集"""
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
