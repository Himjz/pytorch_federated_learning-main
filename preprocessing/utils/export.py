import os
import shutil
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Any, Callable

from .data_splitter import DataSplitter
from .client import Client


class ExportingDataSplitter(DataSplitter):
    """带数据导出功能的数据集划分器"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.export = kwargs.get('export', False)
        self._info_printed = False  # 用于跟踪信息是否已打印

    def divide(self, create: bool = False) -> Any:
        """重写划分方法，在划分后执行导出"""
        result = super().divide(create)

        if self.export and self.is_divided:
            self._export_clients_data()

        return result

    def _export_clients_data(self):
        """导出各客户端的数据集到指定目录结构"""
        export_base = os.path.join(self.root, f"{self.dataset_name}_clients_export")

        if os.path.exists(export_base):
            shutil.rmtree(export_base)

        os.makedirs(export_base, exist_ok=True)

        # 从客户端ID中提取数字部分进行排序（处理f_00000格式）
        def extract_number(client_id):
            try:
                return int(client_id.split('_')[-1])
            except (IndexError, ValueError):
                return 0

        # 使用提取的数字进行排序，保持客户端顺序
        sorted_client_ids = sorted(self.clients.keys(), key=extract_number)
        total_clients = len(sorted_client_ids)

        # 计算需要的数字位数
        num_digits = len(str(total_clients - 1)) if total_clients > 0 else 5

        # 计算总样本数，用于总体进度条
        total_samples = sum(len(client.indices) for client in self.clients.values())

        # 确保先打印所有初始信息
        if not self._info_printed:
            print(f"开始导出 {total_clients} 个客户端的 {total_samples} 个样本...")
            self._info_printed = True  # 标记为已打印

        # 强制刷新缓冲区，确保信息先显示
        import sys
        sys.stdout.flush()

        # 创建总体进度条，现在会在初始信息之后显示
        with tqdm(total=total_samples, desc="整体导出进度", unit="样本",
                  position=0, leave=True, ncols=80,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for idx, client_id in enumerate(sorted_client_ids):
                client = self.clients[client_id]
                formatted_client_name = f"f_{idx:0{num_digits}d}"
                # 导出单个客户端数据并更新进度
                samples_processed = self._export_single_client(formatted_client_name, client, export_base)
                pbar.update(samples_processed)

        # 为内置数据集创建标记文件
        if self._check_if_builtin():
            self._create_builtin_flag(export_base)

        self._generate_export_documents(export_base)
        print(f"\n客户端数据已导出至: {export_base}")

    def _check_if_builtin(self):
        """判断是否为内置数据集"""
        builtin_datasets = {"MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "ImageNet"}
        return self.dataset_name in builtin_datasets

    def _create_builtin_flag(self, export_base):
        """为内置数据集创建标记文件"""
        flag_path = os.path.join(export_base, "flag.txt")
        with open(flag_path, "w", encoding="utf-8") as f:
            f.write(f"original_dataset: {self.dataset_name}\n")
            from datetime import datetime
            f.write(f"export_timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("is_builtin: True\n")

    def _export_single_client(self, client_dir_name: str, client: Client, export_base: str) -> int:
        """导出单个客户端的数据，返回处理的样本数"""
        # 创建客户端目录结构
        client_dir = os.path.join(export_base, client_dir_name)
        train_dir = os.path.join(client_dir, "train")
        val_dir = os.path.join(client_dir, "val")

        # 划分训练集和验证集（8:2比例）
        indices = client.indices
        split_idx = int(0.8 * len(indices))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # 获取客户端拥有的类别
        client_classes = set()
        for idx in indices:
            lbl = self._to_numpy_label(self.base_labels[idx])
            client_classes.add(lbl)

        # 为每个类别创建目录
        for cls in client_classes:
            os.makedirs(os.path.join(train_dir, f"class{cls}"), exist_ok=True)
            os.makedirs(os.path.join(val_dir, f"class{cls}"), exist_ok=True)

        # 导出训练集和验证集，不显示单独进度
        train_count = self._export_samples(train_indices, train_dir)
        val_count = self._export_samples(val_indices, val_dir)

        # 返回总处理样本数
        return train_count + val_count

    def _export_samples(self, indices: List[int], target_dir: str) -> int:
        """导出指定索引的样本到目标目录，返回导出的样本数"""
        for idx, data_idx in enumerate(indices):
            # 获取图像和标签
            if self.base_images is not None:
                img_tensor = self.base_images[data_idx]
            else:
                img_tensor, _ = self.original_trainset[data_idx]

            # 获取标签
            label = self._to_numpy_label(self.base_labels[data_idx])

            # 转换为PIL图像（反归一化处理）
            img = self._tensor_to_pil(img_tensor)

            # 保存路径
            class_dir = os.path.join(target_dir, f"class{label}")
            img_path = os.path.join(class_dir, f"{idx}.png")

            # 保存图像
            img.save(img_path)

        return len(indices)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """将张量转换为PIL图像（处理归一化）"""
        # 反归一化
        if self.in_channels == 1 or self.dataset_name in ['MNIST', 'FashionMNIST']:
            mean = torch.tensor([0.5])
            std = torch.tensor([0.5])
        else:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])

        tensor = tensor * std[:, None, None] + mean[:, None, None]
        tensor = torch.clamp(tensor, 0, 1)

        # 转换为PIL图像
        return transforms.ToPILImage()(tensor)

    def _generate_export_documents(self, export_base: str):
        """生成导出目录的说明文档"""
        # 生成README.md和introduce.md的代码保持不变
        readme_content = """# 客户端数据集目录说明

## 目录结构
本目录包含按联邦学习客户端划分的数据集，每个客户端拥有独立的训练集和验证集。

## 客户端命名规则
客户端目录目录采用统一命名格式：`f_00000`、`f_00001`...，其中数字部分为客户端索引，位数根据客户端总数自动调整。

## 使用规范
- 每个客户端目录下的train文件夹用于模型训练
- 每个客户端目录下的val文件夹用于模型验证
- 图像文件均为PNG格式
- flag.txt文件（如存在）记录原始内置数据集信息，请勿删除或修改

## 维护责任人
自动生成，维护者：系统管理员

## 扩展规则
如需添加新客户端，请保持相同的目录结构和命名规范
"""
        with open(os.path.join(export_base, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)

        class_defs = "\n".join([f"- class{cls}: 对应原始数据集的{cls}类别"
                                for cls in self.selected_classes])

        builtin_note = ""
        if self._check_if_builtin():
            builtin_note = f"""
## 内置数据集转换说明
原始{self.dataset_name}数据集已转换为PNG格式：
- 单通道图像已转换为RGB三通道（如适用）
- 标准化参数已逆转以恢复原始像素值范围
- 图像尺寸已统一调整为{self.image_size}x{self.image_size}

## 恢复信息
flag.txt文件记录了原始数据集名称和导出时间，可用于数据集溯源和恢复
"""

        introduce_content = f"""# 数据集说明文档

## 基本信息
- 原始数据集: {self.dataset_name}
- 客户端数量: {len(self.clients)}
- 总样本数: {sum(len(client) for client in self.clients.values())}
- 图像分辨率: {self.image_size}x{self.image_size}
- 图像格式: PNG
- 是否为内置数据集: {'是' if self._check_if_builtin() else '否'}

## 数据来源
所有数据来源于原始{self.dataset_name}数据集，按{self.distribution_type}分布策略划分

## 客户端命名规则
客户端目录命名格式为`f_00000`，其中数字部分为客户端索引，确保排序一致性。

## 类别定义
{class_defs}

## 数据授权信息
遵循原始{self.dataset_name}数据集的授权协议

## 质量问题记录
无特殊质量问题

{builtin_note}
"""
        with open(os.path.join(export_base, "introduce.md"), "w", encoding="utf-8") as f:
            f.write(introduce_content)
