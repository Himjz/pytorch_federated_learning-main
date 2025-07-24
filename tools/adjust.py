import os
import random
import shutil

from tqdm import tqdm


def adjust_dataset_accuracy(src_root, dst_root, accuracy, exclude_dirs=None):
    """
    调整数据集的标注正确率（仅处理train目录）

    参数:
        src_root (str): 原始数据集根目录
        dst_root (str): 调整后数据集保存目录
        accuracy (float): 目标标注正确率（0-1之间）
        exclude_dirs (list): 不进行调整的目录（默认包含test、val等）
    """
    # 设置默认不调整的目录
    if exclude_dirs is None:
        exclude_dirs = ['test', 'val', 'validation', 'testset', 'valset']

    # 创建目标目录
    os.makedirs(dst_root, exist_ok=True)

    # 获取所有一级目录
    all_dirs = [d for d in os.listdir(src_root)
                if os.path.isdir(os.path.join(src_root, d))]

    # 分离需要处理的目录和直接复制的目录
    process_dirs = [d for d in all_dirs if d not in exclude_dirs]
    copy_dirs = [d for d in all_dirs if d in exclude_dirs]

    # 直接复制无需调整的目录
    print(f"===== 开始复制无需调整的目录 =====")
    for dir_name in copy_dirs:
        src_dir = os.path.join(src_root, dir_name)
        dst_dir = os.path.join(dst_root, dir_name)

        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        print(f"已复制: {dir_name} (文件数: {sum(len(files) for _, _, files in os.walk(src_dir))})")

    # 处理需要调整标注的目录（主要是train）
    print(f"\n===== 开始调整标注（目标正确率: {accuracy:.2f}） =====")
    for dir_name in process_dirs:
        src_train = os.path.join(src_root, dir_name)
        dst_train = os.path.join(dst_root, dir_name)

        # 获取所有类别目录
        classes = [c for c in os.listdir(src_train)
                   if os.path.isdir(os.path.join(src_train, c))]
        if not classes:
            print(f"警告: {dir_name}目录下未找到类别子目录，跳过处理")
            continue

        print(f"\n处理目录: {dir_name}（类别数: {len(classes)}）")

        # 创建目标训练目录及类别子目录
        os.makedirs(dst_train, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(dst_train, cls), exist_ok=True)

        # 统计每个类别的文件数量
        class_files = {}
        total_files = 0
        for cls in classes:
            cls_dir = os.path.join(src_train, cls)
            files = [f for f in os.listdir(cls_dir)
                     if os.path.isfile(os.path.join(cls_dir, f))]
            class_files[cls] = files
            total_files += len(files)
            print(f"  原始类别 {cls}: {len(files)} 个文件")

        # 处理每个类别的文件
        adjusted_files = 0
        for true_cls in tqdm(classes, desc="调整标注进度"):
            files = class_files[true_cls]
            num_files = len(files)
            if num_files == 0:
                continue

            # 计算需要错误标注的文件数量
            num_correct = int(num_files * accuracy)
            num_incorrect = num_files - num_correct

            # 随机选择需要错误标注的文件
            incorrect_files = random.sample(files, num_incorrect)
            correct_files = [f for f in files if f not in incorrect_files]

            # 复制正确标注的文件
            for f in correct_files:
                src_path = os.path.join(src_train, true_cls, f)
                dst_path = os.path.join(dst_train, true_cls, f)
                shutil.copy2(src_path, dst_path)

            # 处理错误标注的文件（复制到其他类别目录）
            for f in incorrect_files:
                # 随机选择一个不同于真实类别的目标类别
                other_classes = [cls for cls in classes if cls != true_cls]
                if not other_classes:
                    # 只有一个类别时无法错误标注，直接复制到原类别
                    dst_cls = true_cls
                else:
                    dst_cls = random.choice(other_classes)

                src_path = os.path.join(src_train, true_cls, f)
                dst_path = os.path.join(dst_train, dst_cls, f)

                # 处理文件名冲突（如果目标路径已存在文件）
                if os.path.exists(dst_path):
                    name, ext = os.path.splitext(f)
                    dst_path = os.path.join(dst_train, dst_cls, f"{name}_dup{ext}")

                shutil.copy2(src_path, dst_path)
                adjusted_files += 1

        print(f"  总文件数: {total_files}")
        print(f"  错误标注文件数: {adjusted_files}")
        print(f"  实际标注正确率: {1 - (adjusted_files / total_files) if total_files > 0 else 1.0:.4f}")

    print(f"\n===== 数据集调整完成 =====")
    print(f"原始数据集: {src_root}")
    print(f"调整后数据集: {dst_root}")
    print(f"目标标注正确率: {accuracy:.2f}")


if __name__ == "__main__":
    # 直接设置参数，无需命令行解析
    src_root = "dt2"  # 原始数据集根目录
    dst_root = "dt3"  # 调整后数据集保存目录
    accuracy = 0  # 目标标注正确率（0-1之间）
    exclude_dirs = None  # 不进行调整的目录（默认包含test、val等）

    # 验证正确率参数
    if not (0 <= accuracy <= 1):
        raise ValueError("标注正确率必须在0到1之间")

    # 执行数据集调整
    adjust_dataset_accuracy(
        src_root=src_root,
        dst_root=dst_root,
        accuracy=accuracy,
        exclude_dirs=exclude_dirs
    )