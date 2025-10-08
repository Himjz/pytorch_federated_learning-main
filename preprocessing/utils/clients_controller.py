import os
import shutil
import sys
from datetime import datetime
from typing import Dict

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from .base_loader import BaseDataLoader
from .client import Client


class ClientsController:
    def __init__(self, class_to_idx, test_set, export,full_train_set,dataset_name,root,in_channels,base_images):
        self.root = root
        self.dataset_name = dataset_name
        self._export = export
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.test_set = test_set
        self.valid_val_labels = []
        self.clients = {}
        self.full_train_set = full_train_set
        self.in_channels = in_channels
        self.base_images = base_images

    def export(self) -> None:
        if self._export:
            self._clean_processed_test_set_indices()
            self._generate_valid_labels()
            self._export_clients_data()

    def _clean_processed_test_set_indices(self):
        test_set__size = len(self.test_set.dataset) if hasattr(self.test_set, 'dataset') else len(
            self.test_set)
        original_indices = self.test_set.indices
        valid_indices = [idx for idx in original_indices if 0 <= idx < test_set__size]

        invalid_count = len(original_indices) - len(valid_indices)
        if invalid_count > 0:
            print(f"警告: 过滤掉 {invalid_count} 个无效的测试集索引")
            self.test_set.indices = valid_indices

    def _generate_valid_labels(self):
        for idx in self.test_set.indices:
            _, label = self.test_set.dataset[idx]
            self.valid_val_labels.append(BaseDataLoader.to_numpy_label(label))

        self.client_valid_labels = {}
        for client_id, client in self.clients.items():
            labels = []
            for idx in client.train_indices:
                _, label = self.full_train_set[idx]
                labels.append(BaseDataLoader.to_numpy_label(label))
            self.client_valid_labels[client_id] = labels

    def _export_clients_data(self):
        export_base = os.path.join(self.root, f"{self.dataset_name}_clients_export")

        if os.path.exists(export_base):
            shutil.rmtree(export_base)
        os.makedirs(export_base, exist_ok=True)

        val_dir = os.path.join(export_base, "global_val")
        os.makedirs(val_dir, exist_ok=True)

        sorted_client_ids = sorted(self.clients.keys(),
                                   key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        total_clients = len(sorted_client_ids)

        total_train_samples = sum(len(client.train_indices) for client in self.clients.values())
        val_count = len(self.test_set.indices)

        print(f"导出 {total_clients} 个客户端数据至: {export_base}")
        print(f"总训练样本: {total_train_samples}, 全局验证样本: {val_count}")
        self._info_printed = True
        sys.stdout.flush()

        print("开始导出全局验证集...")
        with tqdm(total=val_count, desc="导出全局验证集", unit="样本",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            self._export_global_validation_set(val_dir, pbar)
        print("全局验证集导出完成")

        print("开始导出客户端数据...")
        with tqdm(total=total_train_samples, desc="导出客户端数据", unit="样本",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for client_id in sorted_client_ids:
                client = self.clients[client_id]
                client_dir = os.path.join(export_base, client_id)
                os.makedirs(client_dir, exist_ok=True)
                processed = self._export_client_training_data(client, client_dir, client_id)
                pbar.update(processed)
        print("客户端数据导出完成")

        if self._is_builtin_dataset():
            # 传递所需参数到配置文件生成方法
            self._create_core_param_file(
                export_base,
                total_train_samples,
                val_count,
                total_clients
            )
        self._generate_export_documents(export_base, total_clients, total_train_samples,
                                        val_count)

    def _export_global_validation_set(self, val_dir: str, pbar: tqdm) -> None:
        class_indices = set(self.valid_val_labels)
        for cls_idx in class_indices:
            cls_name = self._get_class_name(cls_idx)
            os.makedirs(os.path.join(val_dir, cls_name), exist_ok=True)

        for sample_idx, data_idx in enumerate(self.test_set.indices):
            try:
                img_tensor, label = self.test_set.dataset[data_idx]
                class_idx = BaseDataLoader.to_numpy_label(label)
            except IndexError as e:
                print(f"\n错误: 索引 {data_idx} 访问失败 - {str(e)}")
                continue

            class_name = self._get_class_name(class_idx)
            save_path = os.path.join(val_dir, class_name, f"global_val_{sample_idx}.png")
            try:
                self._tensor_to_pil(img_tensor).save(save_path)
            except Exception as e:
                print(f"\n保存失败 {save_path}: {e}")

            pbar.update(1)

    def _export_client_training_data(self, client: Client, client_dir: str, client_name: str) -> int:
        train_indices = client.train_indices
        total_processed = 0

        class_indices = set(self.client_valid_labels[client_name])
        for cls_idx in class_indices:
            cls_name = self._get_class_name(cls_idx)
            os.makedirs(os.path.join(client_dir, cls_name), exist_ok=True)

        for sample_idx, data_idx in enumerate(train_indices):
            if self.base_images is not None:
                img_tensor = self.base_images[data_idx]
                _, label = self.full_train_set[data_idx]
            else:
                img_tensor, label = self.full_train_set[data_idx]

            class_idx = BaseDataLoader.to_numpy_label(label)

            class_name = self._get_class_name(class_idx)
            save_path = os.path.join(client_dir, class_name, f"{client_name}_train_{sample_idx}.png")
            try:
                self._tensor_to_pil(img_tensor).save(save_path)
                total_processed += 1
            except Exception as e:
                print(f"\n保存失败 {save_path}: {e}")

        return total_processed

    def _is_builtin_dataset(self) -> bool:
        return self.dataset_name in {"MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "ImageNet"}

    def _create_core_param_file(self, export_base: str, total_train: int, total_val: int, num_client: int) -> None:
        """创建核心参数文件，按基本程度排序"""
        param_path = os.path.join(export_base, "core_param.txt")
        with open(param_path, "w", encoding="utf-8") as f:
            # 基础核心参数 - 数据集标识与位置
            f.write(f"dataset_name={self.dataset_name}\n")
            f.write(f"root={self.root}\n")
            f.write(f"seed={getattr(self, 'seed', 'None')}\n")

            # 数据基本属性
            f.write(f"size={getattr(self, 'size', 'None')}\n")
            f.write(f"in_channels={self.in_channels}\n")
            f.write(f"subset={getattr(self, 'subset', 'None')}\n")
            f.write(f"cut={getattr(self, 'cut', 'None')}\n")

            # 联邦学习核心配置
            f.write(f"num_client={num_client}\n")
            f.write(f"num_local_class={getattr(self, 'num_local_class', 'None')}\n")
            f.write(f"distribution={getattr(self, 'distribution', 'None')}\n")

            # 数据处理相关
            f.write(f"augmentation={getattr(self, 'augmentation', 'False')}\n")
            f.write(f"augmentation_params={getattr(self, 'augmentation_params', 'None')}\n")

            # 导出配置
            f.write(f"save={getattr(self, 'save', 'True')}\n")
            f.write(f"untrusted_strategies={getattr(self, 'untrusted_strategies', 'None')}\n")

            # 导出统计信息
            f.write(f"total_training_samples={total_train}\n")
            f.write(f"total_validation_samples={total_val}\n")
            f.write(f"export_timestamp={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def _get_class_name(self, class_idx: int) -> str:
        return self.idx_to_class.get(class_idx, f"class{class_idx}")

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if self.in_channels == 1 or self.dataset_name in ['MNIST', 'FashionMNIST']:
            mean, std = torch.tensor([0.5]), torch.tensor([0.5])
        else:
            mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

        if tensor.ndim == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)

        tensor = tensor * std[:, None, None] + mean[:, None, None]
        return transforms.ToPILImage()(torch.clamp(tensor, 0.0, 1.0))

    def _generate_export_documents(self, export_base: str, total_clients: int, total_train: int,
                                   total_val: int) -> None:
        with open(os.path.join(export_base, "README.md"), "w", encoding="utf-8") as f:
            f.write(f"# {self.dataset_name} 联邦学习数据集\n\n")
            f.write("## 目录结构\n")
            f.write("- 客户端文件夹: 包含各客户端的训练样本\n")
            f.write("- global_val: 全局共享验证集\n\n")
            f.write("## 统计信息\n")
            f.write(f"- 客户端数量: {total_clients}\n")
            f.write(f"- 总训练样本: {total_train}\n")
            f.write(f"- 验证样本: {total_val}\n")
            f.write(f"- 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def append_client(self, client: Client) -> None:
        self.clients[client.client_id] = client

    def get_clients(self) -> Dict[str, Client]:
        return self.clients
