import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def sample_existing_split(original_dir, target_dir, train_ratio=0.2, val_ratio=0.2, seed=42):
    """
    对已划分好的train/val数据集分别进行抽样，test目录则完全复制

    参数:
        original_dir: 原始数据集目录，应包含train和val子目录
        target_dir: 目标数据集目录
        train_ratio: 训练集抽样比例(0-1之间)
        val_ratio: 验证集抽样比例(0-1之间)
        seed: 随机种子，确保结果可复现
    """
    # 设置随机种子
    random.seed(seed)

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 处理训练集
    train_dir = os.path.join(original_dir, "train")
    if os.path.exists(train_dir):
        target_train_dir = os.path.join(target_dir, "train")
        os.makedirs(target_train_dir, exist_ok=True)
        print(f"正在抽样训练集，比例: {train_ratio * 100}%")
        _sample_directory(train_dir, target_train_dir, train_ratio)

    # 处理验证集
    val_dir = os.path.join(original_dir, "val")
    if os.path.exists(val_dir):
        target_val_dir = os.path.join(target_dir, "val")
        os.makedirs(target_val_dir, exist_ok=True)
        print(f"正在抽样验证集，比例: {val_ratio * 100}%")
        _sample_directory(val_dir, target_val_dir, val_ratio)

    # 处理测试集（完全复制）
    test_dir = os.path.join(original_dir, "test")
    if os.path.exists(test_dir):
        target_test_dir = os.path.join(target_dir, "test")
        print("正在复制测试集...")
        _copy_directory(test_dir, target_test_dir)

    print(f"数据集处理完成，结果保存在 {target_dir}")


def _sample_directory(source_dir, target_dir, ratio):
    """辅助函数：对单个目录进行抽样"""
    # 遍历所有类别文件夹
    for class_name in tqdm(os.listdir(source_dir), desc="处理类别"):
        class_dir = os.path.join(source_dir, class_name)

        # 跳过非目录文件
        if not os.path.isdir(class_dir):
            continue

        # 确保目标类别目录存在
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        # 获取当前类别下的所有文件
        all_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # 计算需要抽取的文件数量
        num_files_to_sample = max(1, int(len(all_files) * ratio))  # 至少保留1个样本

        # 随机抽取文件
        sampled_files = random.sample(all_files, num_files_to_sample)

        # 复制抽取的文件到目标目录
        for file_name in sampled_files:
            source_path = os.path.join(class_dir, file_name)
            target_path = os.path.join(target_class_dir, file_name)
            shutil.copy2(source_path, target_path)


def _copy_directory(source_dir, target_dir):
    """辅助函数：完全复制目录"""
    # 如果目标目录已存在，先删除
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # 复制整个目录树
    shutil.copytree(source_dir, target_dir)

    print(f"已完全复制 {source_dir} 到 {target_dir}")


if __name__ == "__main__":
    # 设置参数
    ORIGINAL_DIR = "dt1"  # 原始数据集路径（应包含train/val/test子目录）
    TARGET_DIR = "dt2"  # 目标数据集路径
    TRAIN_RATIO = 0.3  # 训练集抽样比例
    VAL_RATIO = 0.3  # 验证集抽样比例
    SEED = 235235  # 随机种子

    # 执行抽样和复制
    sample_existing_split(ORIGINAL_DIR, TARGET_DIR, TRAIN_RATIO, VAL_RATIO, SEED)