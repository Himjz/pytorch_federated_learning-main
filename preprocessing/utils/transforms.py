import torch
import torchvision.transforms as transforms
from typing import List, Dict, Optional, Any


class TransformMixin:
    """数据转换混合类，提供数据增强和转换功能"""
    
    def __init__(self, augmentation: bool = False, augmentation_params: Optional[Dict[str, Any]] = None):
        self.augmentation = augmentation
        self.augmentation_params = augmentation_params or {
            'rotation': 15,
            'translation': 0.1,
            'scale': 0.1,
            'flip': True
        }
    
    def _get_augmentation_transform(self) -> List[transforms.transforms]:
        """获取数据增强转换列表"""
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
        """创建通道转换，用于调整图像通道数"""
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
        """向转换列表添加标准化步骤"""
        if self.in_channels == 1 or (self.in_channels is None and self.dataset_name in ['MNIST', 'FashionMNIST']):
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
