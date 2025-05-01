import os
import shutil
import random


def split_data(source_dir, train_dir, test_dir, split_ratio=0.8):
    """
    此函数用于将源目录中的数据按照指定比例划分为训练集和测试集
    :param source_dir: 源数据目录，该目录下每个子目录代表一个类别
    :param train_dir: 训练集目标目录
    :param test_dir: 测试集目标目录
    :param split_ratio: 训练集所占比例，默认为 0.8
    """
    # 确保训练集和测试集目录存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历源目录下的每个类别文件夹
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if os.path.isdir(class_dir):
            # 创建训练集和测试集中对应的类别文件夹
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # 获取该类别下的所有图片文件
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                      if os.path.isfile(os.path.join(class_dir, img))]

            # 打乱图片顺序
            random.shuffle(images)

            # 计算训练集和测试集的分割点
            split_index = int(len(images) * split_ratio)

            # 复制图片到训练集和测试集目录
            for i, img in enumerate(images):
                if i < split_index:
                    shutil.copy2(img, train_class_dir)
                else:
                    shutil.copy2(img, test_class_dir)


if __name__ == "__main__":
    # 请根据实际情况修改这些目录路径
    source_directory = '../Data'
    train_directory = 'train'
    test_directory = 'val'
    # 调用函数进行数据划分
    split_data(source_directory, train_directory, test_directory)