from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

def split_dataset(data_dir, train_ratio=0.8, transform=transform):
    """
    按给定比例划分数据集为训练集和测试集。

    Args:
        data_dir (str): 数据集根目录，该目录下直接包含各类别的文件夹
        train_ratio (float, optional): 训练集所占比例，默认 0.8
        transform (callable, optional): 数据预处理转换操作，默认使用上面定义的 transform

    Returns:
        tuple: 包含训练集数据加载器和测试集数据加载器的元组
    """
    # 从根目录加载完整数据集
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # 计算训练集大小
    train_size = int(train_ratio * len(full_dataset))
    # 计算测试集大小
    test_size = len(full_dataset) - train_size
    # 划分数据集
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 创建训练集数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # 创建测试集数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


# 划分数据集并获取数据加载器
train_loader, test_loader = split_dataset('..\\Data', 0.8)
print(train_loader.targets)
# 收集训练集标签
train_labels = []
for _, labels in train_loader:
    train_labels.extend(labels.tolist())

# 收集测试集标签
test_labels = []
for _, labels in test_loader:
    test_labels.extend(labels.tolist())

# 打印所有标签
print("训练集标签:", train_labels)
print("测试集标签:", test_labels)

# 示例：遍历训练集
for images, labels in train_loader:
    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")