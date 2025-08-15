# 联邦学习数据加载器 (v5.0.1) 说明文档

## 概述

联邦学习数据加载器（UniversalDataLoader v5.0.1）是一个高度可定制的工具，用于为联邦学习场景准备和分配数据集。该加载器支持多种标准数据集和自定义数据集，能够模拟不同的客户端数据分布，并支持各种客户端行为策略（包括恶意行为）。

## 核心接口

### 主要类：`UniversalDataLoader`

该类是数据加载器的核心接口，继承自以下组件：
- `BaseDataLoader`：提供基础属性和通用方法
- `TransformMixin`：提供数据增强和转换功能
- `DatasetLoader`：处理数据集加载逻辑
- `DataSplitter`：负责数据集划分到客户端

### 初始化参数

| 参数名 | 类型 | 描述 | 默认值 |
|--------|------|------|--------|
| `root` | str | 数据集存储根目录 | './data' |
| `dataset_name` | str | 数据集名称（支持'MNIST'、'CIFAR10'等或自定义路径） | 'MNIST' |
| `num_client` | int | 客户端数量 | 1 |
| `num_local_class` | int | 每个客户端拥有的类别数量 | 10 |
| `untrusted_strategies` | List[float] | 客户端策略列表，决定客户端行为 | None |
| `k` | int | 类别分配参数 | 1 |
| `print_report` | bool | 是否打印数据集报告 | True |
| `device` | torch.device | 计算设备 | 自动检测（GPU优先） |
| `preload_to_gpu` | bool | 是否将数据预加载到GPU | False |
| `seed` | int | 随机种子，保证结果可复现 | 0 |
| `download` | bool | 是否下载数据集 | True |
| `cut` | float | 样本裁剪比例（0-1之间） | None |
| `subset` | Tuple[str, Union[int, float]] | 类别筛选参数 | None |
| `size` | Union[int, Tuple[int, int]] | 图像尺寸 | None |
| `in_channels` | int | 输入图像通道数 | None |
| `augmentation` | bool | 是否启用数据增强 | False |
| `augmentation_params` | Dict[str, Any] | 数据增强参数 | 见下文默认值 |

**默认增强参数**：
```python
{
    'rotation': 15,        # 旋转角度
    'translation': 0.1,    # 平移比例
    'scale': 0.1,          # 缩放比例
    'flip': True           # 是否水平翻转
}
```

### 主要方法

| 方法名 | 描述 | 返回值 |
|--------|------|--------|
| `load()` | 加载并预处理数据集 | 自身实例 |
| `divide(create=False)` | 将数据集划分给客户端 | 若create=True返回自身，否则返回(trainset_config, testset) |
| `calculate_statistics(sample_size=1000)` | 计算数据集的均值和标准差 | (mean, std) |
| `get_test_loader(batch_size=32, shuffle=False)` | 获取测试集数据加载器 | DataLoader |

## 使用示例

```python
import torch
from fed_dataloader import UniversalDataLoader

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 初始化数据加载器
loader = UniversalDataLoader(
    root="..",
    num_client=5,
    num_local_class=2,
    dataset_name='dt',
    seed=0,
    untrusted_strategies=[2.1, 3.5, 4.3, 0.5, 1.0],
    device=device,
    preload_to_gpu=False,
    in_channels=1,
    size=32,
    augmentation=True,
    cut=0.8,  # 保留80%的样本
    subset=('random', 3)  # 随机选择3个类别
)

# 加载数据集
loader.load()

# 计算数据集统计信息
mean, std = loader.calculate_statistics()
print(f"\n数据集统计信息: 均值={mean}, 标准差={std}")

# 划分数据集到客户端
trainset_config, testset = loader.divide()

# 打印数据集信息
print("\n数据集信息:")
print(f"  类别数: {loader.num_classes}")
print(f"  图像尺寸: {loader.image_size}x{loader.image_size}")
print(f"  图像通道数: {loader.in_channels}")

# 验证客户端加载器
for i in range(3):
    client_id = f"f_0000{i}"
    if client_id in trainset_config['user_data']:
        trainset_config['user_data'][client_id].verify_loader()
    else:
        print(f"客户端 {client_id} 不存在")
```

## 客户端策略说明

`untrusted_strategies`参数接受浮点数值列表，每个值定义一个客户端的行为策略：

- **正常客户端** (1.0)：正常使用数据，不做任何篡改
- **恶意客户端** (0.x)：篡改部分标签，x表示标签准确率（0.5表示50%标签正确）
- **稀疏客户端** (2.x)：仅使用部分数据，x表示使用比例（0.1表示使用10%数据）
- **特征噪声客户端** (3.x)：在特征中添加噪声，x表示噪声强度
- **类别倾斜客户端** (4.x)：某些类别的样本比例更高，x表示倾斜程度

## 扩展指南

### 1. 添加新的数据集

要支持新的标准数据集，可在`utils/dataset_loader.py`中的`default_datasets`字典添加条目：

```python
# 在DatasetLoader类中
self.default_datasets = {
    # 现有条目...
    'NEW_DATASET': (NewDatasetClass, num_classes),
}
```

### 2. 实现新的客户端策略

1. 在`utils/client.py`的`Client`类中添加新的策略处理逻辑：

```python
def apply_strategy(self) -> None:
    # 现有策略...
    elif self.strategy == 'new_strategy':
        # 实现新策略逻辑
        self.new_strategy_param = self.strategy_ratio
        # ...
```

2. 在`utils/data_splitter.py`的`parse_strategy`方法中添加新策略解析：

```python
@staticmethod
def parse_strategy(strategy_code: Union[float, str]) -> Tuple[str, float]:
    try:
        code = float(strategy_code)
        integer_part = int(code)
        decimal_part = code - integer_part

        strategies = {
            # 现有策略...
            5: ('new_strategy', decimal_part),  # 添加新策略
        }

        return strategies.get(integer_part, ('normal', 1.0))
    except (ValueError, TypeError):
        return 'normal', 1.0
```

### 3. 添加新的数据增强方法

在`utils/transforms.py`的`_get_augmentation_transform`方法中添加新的变换：

```python
def _get_augmentation_transform(self) -> List[transforms.transforms]:
    aug_transforms = []
    # 现有增强...
    
    # 添加新的增强
    if self.augmentation_params.get('new_aug', False):
        aug_transforms.append(transforms.RandomSomething(...))
    
    return aug_transforms
```

### 4. 自定义数据划分逻辑

创建`DataSplitter`的子类，重写数据划分相关方法：

```python
class CustomDataSplitter(DataSplitter):
    def _assign_classes_to_clients(self) -> Dict[str, List[int]]:
        # 实现自定义的类别分配逻辑
        pass
        
    def _create_clients(self, client_classes, class_indices, class_assign_counts):
        # 实现自定义的客户端创建逻辑
        pass
```

然后在`fed_dataloader.py`中让`UniversalDataLoader`继承自这个新类。

## 版本历史

v5.0.1:
- 优化了部分BUG

v5.0.0:
- 采用模块化设计，拆分代码为多个文件
- 实现链式继承结构，提高代码复用性
- 统一接口设计，保持与原有测试代码兼容
- 增强了自定义数据集的支持

v4.2.0:
- 新增划分子数据集功能

v4.1.0:
- 新增了数据增强功能
- 修复了数据加载时的一些问题

v4.0.0:
- 将加载器对象化，修改了接口

v3.0.0:
- 将两个数据加载器合并 

v2.0.0:
- 为自定义数据加载器添加不可信客户端构建功能

v1.0.0:
- 新增自定义数据加载器，加载自定义数据
