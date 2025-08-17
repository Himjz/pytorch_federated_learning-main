import random
from typing import Tuple, Union

import numpy as np
import torch


class BaseDataLoader:
    """基础数据加载器类，提供核心属性和通用方法"""

    def __init__(self, **kwargs):
        # 数据集基本信息
        self.dataset_name = kwargs.get('dataset_name', 'MNIST')
        self.root = kwargs.get('root', './data')
        self.download = kwargs.get('download', True)
        self.seed = kwargs.get('seed', 0)

        # 联邦学习相关参数
        self.num_client = kwargs.get('num_client', 1)
        self.num_local_class = kwargs.get('num_local_class', 10)
        self.untrusted_strategies = kwargs.get('untrusted_strategies', None)
        distribution = kwargs.get('distribution', ('default', 1))
        self.distribution_type, self.distribution_param = distribution

        # 数据处理参数
        self.cut_ratio = kwargs.get('cut', None)
        self.subset_params = kwargs.get('subset', None)
        self.image_size = kwargs.get('size', None)
        self.in_channels = kwargs.get('in_channels', None)
        self.augmentation = kwargs.get('augmentation', False)
        self.augmentation_params = kwargs.get('augmentation_params', None)

        # 设备和性能参数
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.preload_to_gpu = kwargs.get('preload_to_gpu', False)
        self.print_report = kwargs.get('print_report', True)
        self.save = kwargs.get('save', False)

        # 内部状态变量
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
        self.processed_trainset = None  # 处理后的训练集（应用了subset和cut）
        self.processed_testset = None  # 处理后的测试集（应用了subset和cut）

        self.is_loaded = False
        self.is_divided = False

        # 设置随机种子
        self.set_seed(self.seed)

    def set_seed(self, seed: int) -> None:
        """设置随机种子以保证结果可复现"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def calculate_statistics(self, sample_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算数据集的均值和标准差"""
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

    def get_test_loader(self, batch_size: int = 32, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """获取测试集数据加载器"""
        if not self.is_loaded:
            raise RuntimeError("请先调用load()方法加载数据集")

        if not self.testset:
            raise ValueError("没有可用的测试集")

        return torch.utils.data.DataLoader(
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
        return len(self.indices) if self.indices is not None else 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        """获取单个样本"""
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
