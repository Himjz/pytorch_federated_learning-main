# 联邦学习数据加载器 (v5.2.2) 说明文档

## 概述

联邦学习数据加载器（UniversalDataLoader v5.2.2）是一款专为联邦学习场景设计的高度可定制化工具，专注于数据集的准备、处理与分布式分配。该工具支持多种标准数据集与自定义数据集加载，能够灵活模拟不同客户端的数据分布特征，并支持多样化的客户端行为策略（包括恶意行为模拟），通过模块化架构设计确保了良好的扩展性与易用性。

## 核心架构与接口

### 主要类：`UniversalDataLoader`

作为数据加载器的核心接口，该类通过多层继承实现功能聚合，继承关系如下：

- `BaseDataLoader`：提供基础属性定义与通用工具方法（如随机种子设置、统计计算等）
- `TransformMixin`：封装数据增强与格式转换逻辑
- `DatasetLoader`：处理具体的数据集加载与预处理流程
- `DataSplitter`：实现数据集向多个客户端的划分逻辑
- 内置客户端控制器：管理客户端生命周期与数据导出功能

### 初始化参数

| 参数名                    | 类型                            | 描述                                                                                               | 默认值            |
|------------------------|-------------------------------|--------------------------------------------------------------------------------------------------|----------------|
| `root`                 | str                           | 数据集存储根目录                                                                                         | './data'       |
| `dataset_name`         | str                           | 数据集名称（支持'MNIST'、'CIFAR10'、'CIFAR100'、'FashionMNIST'等标准数据集或自定义路径）                       | 'MNIST'        |
| `num_client`           | int                           | 客户端数量                                                                                            | 1              |
| `num_local_class`      | int                           | 每个客户端分配的类别数量                                                                                   | 10             |
| `untrusted_strategies` | List[float]                   | 客户端行为策略列表，每个值定义对应客户端的行为模式                                                                  | None           |
| `distribution`         | Tuple[str, Union[int, float]] | 数据分布策略：<br>- 'default'：轮询分配策略，参数为k值<br>- 'dirichlet'：狄利克雷分布（Non-IID），参数为alpha值（越小分布越不均衡） | ('default', 1) |
| `print_report`         | bool                          | 是否打印数据集加载与划分的详细报告                                                                              | True           |
| `device`               | torch.device                  | 计算设备（CPU/GPU）                                                                                     | 自动检测（GPU优先）    |
| `preload_to_gpu`       | bool                          | 是否将数据预加载到GPU内存                                                                                  | False          |
| `seed`                 | int                           | 随机种子，保证实验可复现                                                                                   | 0              |
| `download`             | bool                          | 是否自动下载标准数据集                                                                                     | True           |
| `cut`                  | float                         | 样本裁剪比例（0-1之间），保留指定比例的样本                                                                       | None           |
| `subset`               | Tuple[str, Union[int, float]] | 类别筛选参数，用于从数据集中选择部分类别                                                                         | None           |
| `size`                 | Union[int, Tuple[int, int]]   | 图像尺寸（宽×高），自动调整数据集图像大小                                                                         | None           |
| `in_channels`          | int                           | 输入图像通道数（1为灰度图，3为RGB图）                                                                           | None           |
| `augmentation`         | bool                          | 是否启用数据增强                                                                                         | False          |
| `export`               | bool                          | 是否导出划分后的客户端数据集与全局验证集                                                                         | False          |
| `augmentation_params`  | Dict[str, Any]                | 数据增强参数配置                                                                                         | 见下文默认值         |
| `save`                 | bool                          | 是否保存处理后的数据集                                                                                     | False          |

**默认增强参数**：

```python
{
    'rotation': 15,  # 随机旋转角度范围（-15°至15°）
    'translation': 0.1,  # 随机平移比例（宽度和高度的10%）
    'scale': 0.1,  # 随机缩放比例（0.9至1.1倍）
    'flip': True  # 是否启用水平随机翻转
}
```

### 主要方法

| 方法名                                             | 描述                                                                 | 返回值                                             |
|-------------------------------------------------|--------------------------------------------------------------------|-------------------------------------------------|
| `load()`                                        | 加载数据集并执行预处理（包括格式转换、尺寸调整、数据增强等）                             | 自身实例（支持链式调用）                                  |
| `divide(create=False)`                          | 将预处理后的数据集按配置划分给客户端，支持自动导出（需开启export参数）                       | 若create=True返回自身，否则返回(trainset_config, testset) |
| `calculate_statistics(sample_size=1000)`        | 随机采样计算数据集的均值和标准差，用于数据标准化                                        | (mean, std) 元组                                 |
| `get_test_loader(batch_size=32, shuffle=False)` | 创建测试集的数据加载器，支持指定批次大小和是否打乱顺序                                     | torch.utils.data.DataLoader 实例                 |
| `verify_loader()` (客户端方法)                  | 验证客户端数据加载器的有效性，检查数据形状、设备分配和标签一致性                             | 无返回值，打印验证结果                                 |

## 使用示例

```python
import torch
from preprocessing.fed_dataloader import UniversalDataLoader

# 设置计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 初始化数据加载器
loader = UniversalDataLoader(
    root="../data",
    num_client=20,
    num_local_class=5,
    dataset_name='MNIST',
    seed=0,
    untrusted_strategies=[2.1, 3.5, 4.3, 0.5, 1.0],  # 客户端策略列表
    device=device,
    preload_to_gpu=False,
    in_channels=1,
    size=32,
    augmentation=True,
    cut=0.8,  # 保留80%的样本
    distribution=('dirichlet', 0.3),  # 使用狄利克雷分布
    export=True  # 启用数据导出
)

# 加载并预处理数据集
loader.load()

# 计算数据集统计信息
mean, std = loader.calculate_statistics()
print(f"\n数据集统计信息: 均值={mean}, 标准差={std}")

# 划分数据集到客户端
trainset_config, testset = loader.divide()

# 打印数据集基本信息
print("\n数据集信息:")
print(f"  类别数: {loader.num_classes}")
print(f"  图像尺寸: {loader.image_size}x{loader.image_size}")
print(f"  图像通道数: {loader.in_channels}")

# 验证客户端数据加载器
for i in range(3):
    client_id = f"f_0000{i}"
    if client_id in trainset_config['user_data']:
        trainset_config['user_data'][client_id].verify_loader()
    else:
        print(f"客户端 {client_id} 不存在")
```

## 客户端策略说明

`untrusted_strategies`参数通过浮点数值定义客户端行为，整数部分表示策略类型，小数部分表示策略参数：

- **正常客户端** (1.0)：使用原始数据，不做任何修改
- **恶意客户端** (0.x)：篡改部分标签，x为标签保留准确率（如0.5表示50%标签正确）
- **稀疏客户端** (2.x)：仅使用部分训练数据，x为数据保留比例（如0.1表示使用10%数据）
- **特征噪声客户端** (3.x)：在图像特征中添加高斯噪声，x为噪声强度（值越大噪声越强）
- **类别倾斜客户端** (4.x)：使某些类别样本比例偏高，x为倾斜程度（值越大倾斜越明显）

## 数据导出说明

当`export=True`时，系统会自动导出以下内容至`root/{dataset_name}_clients_export`目录：

- **客户端数据**：每个客户端的训练样本按类别存储在对应子目录
- **全局验证集**：存储在`global_val`目录，按类别划分
- **核心参数文件**（`core_param.txt`）：记录数据集配置、划分参数和统计信息
- **说明文档**（`README.md`）：包含目录结构和关键统计信息

导出过程会自动处理无效索引，并通过进度条显示导出进度，支持大型数据集的分批处理。

## 扩展指南

### 1. 添加新的标准数据集

在`preprocessing/utils/dataset_loader.py`的`DatasetLoader`类中，向`default_datasets`字典添加条目：

```python
self.default_datasets = {
    # 现有条目...
    'NEW_DATASET': (NewDatasetClass, num_classes),  # 新增数据集
}
```

其中`NewDatasetClass`为数据集处理类（需兼容torchvision数据集接口），`num_classes`为该数据集的类别数量。

### 2. 实现新的客户端策略

1. 在`preprocessing/utils/base_loader.py`的`parse_strategy`方法中添加策略解析：

```python
strategies = {
    # 现有策略...
    5: ('new_strategy', decimal_part),  # 新增策略映射
}
```

2. 在`preprocessing/utils/client.py`的`Client`类中实现策略逻辑：

```python
def apply_strategy(self):
    # 现有策略处理...
    elif self.strategy == 'new_strategy':
        # 实现新策略的具体逻辑
        self.processed_data = self._apply_new_strategy()
```

### 3. 添加新的数据增强方法

在`preprocessing/utils/transforms.py`的`_get_augmentation_transform`方法中扩展：

```python
def _get_augmentation_transform(self) -> List[transforms.transforms]:
    aug_transforms = []
    # 现有增强处理...
    
    # 新增增强方法
    if self.augmentation_params.get('custom_aug', False):
        aug_transforms.append(transforms.RandomCustomAugmentation(...))
    
    return aug_transforms
```

### 4. 自定义数据划分逻辑

创建`DataSplitter`的子类并重写关键方法：

```python
from preprocessing.utils.data_splitter import DataSplitter

class CustomDataSplitter(DataSplitter):
    def _assign_classes_to_clients(self, valid_classes):
        # 自定义客户端-类别分配逻辑
        pass
    
    def _create_clients(self, client_classes, class_indices, class_assign_counts):
        # 自定义客户端数据分配逻辑
        pass
```

然后在`preprocessing/fed_dataloader.py`中更新`UniversalDataLoader`的继承关系。

## 版本历史

v5.2.2:

- 新增客户端控制器（`ClientsController`），统一管理客户端生命周期
- 重构数据导出逻辑，整合至客户端控制器
- 移除独立导出器，优化导出性能

v5.2.1:

- 修复数据集导出时的索引越界问题
- 完善导出文件的元数据记录
- 新增导出过程的错误处理与日志输出

v5.2.0:

- 新增数据集导出功能，支持客户端数据与全局验证集导出
- 优化导出文件的目录结构，按类别组织样本
- 添加导出统计信息与说明文档自动生成

v5.1.0:

- 新增狄利克雷（Dirichlet）分布支持，用于构建Non-IID数据集
- 增强数据分布报告，显示类别分配详情
- 优化随机种子管理，确保划分结果可复现

v5.0.0:

- 采用模块化设计，拆分代码为多个功能文件
- 实现链式继承结构，提升代码复用率
- 统一接口设计，保持向下兼容性
- 增强自定义数据集支持，优化加载流程

v4.2.0:

- 新增划分子数据集功能（`subset`参数）
- 优化类别筛选逻辑

v4.1.0:

- 新增数据增强模块
- 修复数据加载中的格式兼容问题

v4.0.0:

- 重构为面向对象设计
- 优化核心接口，简化使用流程

v3.0.0:

- 合并标准数据集与自定义数据集加载器
- 统一数据处理流程

v2.0.0:

- 为自定义数据加载器添加不可信客户端支持

v1.0.0:

- 初始版本，支持自定义数据集加载