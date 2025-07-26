import os
import shutil
from collections import Counter
from typing import List, Dict, Optional
from PIL import Image


class TopNClassSubsetBuilder:
    """通过复制图片构建top-n类子集的工具类"""

    def __init__(self, root_dir: str):
        """
        初始化工具类

        参数:
            root_dir: 原始数据集根目录，需包含'train'和'val'子文件夹
        """
        self.root_dir = root_dir
        self.split_dirs = {  # 训练集和验证集路径
            'train': os.path.join(root_dir, 'train'),
            'val': os.path.join(root_dir, 'val')
        }

        # 验证数据集结构
        for split, path in self.split_dirs.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"未找到{split}目录: {path}")

        # 存储每个split的类别信息
        self.classes = {split: self._get_classes(split) for split in ['train', 'val']}
        self.class_counts = {split: self._count_samples(split) for split in ['train', 'val']}

    def _get_classes(self, split: str) -> List[str]:
        """获取指定split（train/val）下的所有类别名称"""
        split_dir = self.split_dirs[split]
        # 类别为split目录下的子文件夹名称
        classes = [d for d in os.listdir(split_dir)
                   if os.path.isdir(os.path.join(split_dir, d))]
        return sorted(classes)  # 按名称排序，确保一致性

    def _count_samples(self, split: str) -> Dict[str, int]:
        """统计指定split下每个类别的样本数量"""
        split_dir = self.split_dirs[split]
        class_counts = {}
        for cls in self.classes[split]:
            cls_dir = os.path.join(split_dir, cls)
            # 统计该类别下的图片文件数量
            img_count = len([f for f in os.listdir(cls_dir)
                             if os.path.isfile(os.path.join(cls_dir, f))])
            class_counts[cls] = img_count
        return class_counts

    def print_class_distribution(self, split: str = 'train', title: str = "类别分布统计"):
        """打印指定split的类别分布（按样本数降序）"""
        if split not in ['train', 'val']:
            raise ValueError("split必须为'train'或'val'")

        print(f"\n===== {title} =====")
        class_counts = self.class_counts[split]
        total = sum(class_counts.values())
        print(f"数据集: {split}，总样本数: {total}")
        print("类别（按样本数降序）:")
        # 按样本数降序排序并打印
        for cls, cnt in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {cnt}张图片（占比{cnt / total:.2%}）")

    def select_top_n_classes(self, n: int, split: str = 'train') -> List[str]:
        """
        选择指定split中样本量最多的top-n个类别

        参数:
            n: 要选择的类别数量
            split: 基于哪个split统计（train/val）
        返回:
            top-n类别名称列表
        """
        if split not in ['train', 'val']:
            raise ValueError("split必须为'train'或'val'")
        if n <= 0 or n > len(self.classes[split]):
            raise ValueError(f"n必须为1到{len(self.classes[split])}之间的整数")

        # 按样本数降序排序，取前n个类别
        sorted_classes = sorted(
            self.class_counts[split].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [cls for cls, _ in sorted_classes[:n]]

    def build_subset(self, n: int, target_root: str, include_val: bool = True,
                     overwrite: bool = False):
        """
        构建包含top-n类别的子集（通过复制图片）

        参数:
            n: 要包含的top-n类别数量
            target_root: 子集保存的根目录
            include_val: 是否同时处理val集（否则只处理train集）
            overwrite: 若目标目录已存在，是否覆盖（False则跳过已存在文件）
        """
        # 1. 选择top-n类别（基于train集的分布，确保train和val一致）
        top_n_classes = self.select_top_n_classes(n, split='train')
        print(f"\n选中的top-{n}类别: {top_n_classes}")

        # 2. 处理train集和val集
        splits_to_process = ['train'] + (['val'] if include_val else [])
        for split in splits_to_process:
            print(f"\n正在处理{split}集...")
            src_split_dir = self.split_dirs[split]  # 源目录
            tgt_split_dir = os.path.join(target_root, split)  # 目标目录

            # 3. 复制每个top-n类别的图片
            for cls in top_n_classes:
                # 跳过该split中不存在的类别（理论上train和val类别应一致）
                if cls not in self.classes[split]:
                    print(f"警告：{split}集中不存在类别{cls}，已跳过")
                    continue

                src_cls_dir = os.path.join(src_split_dir, cls)  # 源类别目录
                tgt_cls_dir = os.path.join(tgt_split_dir, cls)  # 目标类别目录
                os.makedirs(tgt_cls_dir, exist_ok=True)  # 创建目标目录

                # 获取源目录下的所有图片文件
                img_files = [f for f in os.listdir(src_cls_dir)
                             if os.path.isfile(os.path.join(src_cls_dir, f))]
                if not img_files:
                    print(f"警告：{src_cls_dir}中无图片文件")
                    continue

                # 复制图片
                copied = 0
                skipped = 0
                for img in img_files:
                    src_path = os.path.join(src_cls_dir, img)
                    tgt_path = os.path.join(tgt_cls_dir, img)

                    # 检查是否需要复制
                    if os.path.exists(tgt_path) and not overwrite:
                        skipped += 1
                        continue
                    # 复制图片（保留元数据）
                    shutil.copy2(src_path, tgt_path)
                    copied += 1

                print(f"  类别{cls}: 复制{copied}张，跳过{skipped}张（已存在）")

            # 打印该split的处理结果
            tgt_total = sum(len(os.listdir(os.path.join(tgt_split_dir, cls)))
                            for cls in top_n_classes if os.path.exists(os.path.join(tgt_split_dir, cls)))
            print(f"{split}集处理完成，共复制{tgt_total}张图片到{tgt_split_dir}")

        print(f"\n子集构建完成，保存路径: {os.path.abspath(target_root)}")


# 示例用法
if __name__ == "__main__":
    # 1. 初始化工具（指定原始数据集根目录）
    dataset_root = "./dt1"  # 替换为你的数据集根目录
    builder = TopNClassSubsetBuilder(root_dir=dataset_root)

    builder.print_class_distribution(split='train', title="原始训练集类别分布")

    builder.build_subset(
        n=12,
        target_root="./dt2",  # 子集保存路径
        include_val=True,  # 同时处理val集
        overwrite=False  # 不覆盖已存在文件
    )