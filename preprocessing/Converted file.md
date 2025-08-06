# UniversalDataLoader 文档

## 1. 代码用途

`UniversalDataLoader` 是一个专为联邦学习场景设计的通用数据集加载与处理工具类，主要功能如下：

- 支持多种主流公开数据集（如MNIST、CIFAR10/100、FashionMNIST等）的自动加载与处理
- 兼容自定义数据集，只需按照特定目录结构组织数据
- 提供完整的数据预处理流程，包括尺寸调整、通道转换、归一化等
- 支持数据增强，可配置旋转、平移、缩放、翻转等增强操作
- 实现联邦学习中数据集的智能划分，将数据分配给多个客户端
- 支持多种客户端数据分布策略，如数据稀疏、类别倾斜、特征噪声、恶意标签等，可模拟不同的联邦学习场景
- 提供数据统计计算、结果保存、加载验证等辅助功能

该类的设计目标是简化联邦学习实验中的数据准备工作，支持快速构建多样化的联邦学习数据场景。

## 2. 参数说明

### 2.1 核心参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `dataset_name` | str | 'MNIST' | 数据集名称，可为内置数据集（如'MNIST'）或自定义数据集目录名 |
| `num_client` | int | 1 | 客户端数量 |
| `num_local_class` | int | 10 | 每个客户端分配的类别数量 |
| `untrusted_strategies` | Optional[List[float]] | None | 客户端数据策略列表，用于指定各客户端的数据分布特性 |
| `k` | int | 1 | 类别分配步长，用于客户端类别分配算法 |
| `print_report` | bool | True | 是否打印数据集划分报告 |
| `device` | Optional[torch.device] | None | 数据加载设备，默认自动检测（优先GPU） |
| `preload_to_gpu` | bool | False | 是否将数据预加载到GPU内存 |
| `seed` | int | 0 | 随机种子，确保实验可重复性 |
| `root` | str | './data' | 数据存储根目录 |
| `download` | bool | True | 是否自动下载数据集（仅对内置数据集有效） |
| `cut` | Optional[float] | None | 数据裁剪比例（0-1之间），用于减少数据集规模 |
| `save` | bool | False | 是否保存处理后的数据集 |
| `subset` | Optional[Tuple[str, Union[int, float]]] | None | 类别子集选择参数，格式为(类型, 值)，如('top', 5)表示选择前5类 |
| `size` | Optional[Union[int, Tuple[int, int]]] | None | 图像尺寸，可指定整数（如32表示32x32）或元组（如(32, 64)） |
| `in_channels` | Optional[int] | None | 输入图像通道数，自动检测或手动指定（1为灰度，3为RGB） |
| `augmentation` | bool | False | 是否启用数据增强 |
| `augmentation_params` | Optional[Dict[str, Any]] | None | 数据增强参数配置 |

### 2.2 数据增强参数（`augmentation_params`）

默认参数配置：
```python
{
    'rotation': 15,        # 旋转角度范围（-15到15度）
    'translation': 0.1,    # 平移比例（相对于图像尺寸）
    'scale': 0.1,          # 缩放比例（0.9到1.1倍）
    'flip': True           # 是否启用水平翻转
}
```

### 2.3 客户端策略参数（`untrusted_strategies`）

策略值为浮点数，整数部分表示策略类型，小数部分表示策略参数：
- `0.x`：恶意客户端策略（标签翻转），x表示标签准确率（1-x为翻转比例）
- `2.x`：稀疏数据策略，x表示数据保留比例
- `3.x`：特征噪声策略，x表示噪声强度
- `4.x`：类别倾斜策略，x表示倾斜程度
- 其他值：默认策略（均匀分布）

## 3. 主要方法

| 方法名 | 功能描述 |
|--------|----------|
| `__init__` | 初始化参数设置，配置数据集和联邦学习相关参数 |
| `load()` | 加载数据集，执行数据预处理，返回自身实例 |
| `divide()` | 划分联邦数据集，为每个客户端分配数据，返回数据集配置和测试集 |
| `calculate_statistics()` | 计算数据集的均值和标准差，用于数据归一化 |
| `get_test_loader()` | 获取测试集的数据加载器（DataLoader） |
| `set_seed()` | 设置随机种子，确保实验可重复性 |

## 4. 客户端类（Client）

`Client`类是`UniversalDataLoader`的内部类，代表联邦学习中的一个客户端，主要方法包括：

| 方法名 | 功能描述 |
|--------|----------|
| `get_data_loader()` | 获取客户端数据的数据加载器 |
| `calculate_label_accuracy()` | 计算标签准确率（主要用于恶意客户端） |
| `print_distribution()` | 打印客户端的数据分布情况 |
| `verify_loader()` | 验证客户端数据加载器的正确性 |
| `__len__` | 返回客户端数据样本数量 |
| `__getitem__` | 获取客户端的一个数据样本 |

## 5. 使用方法

### 5.1 基本使用流程

1. 创建`UniversalDataLoader`实例并配置参数
2. 调用`load()`方法加载数据集
3. 调用`divide()`方法划分联邦数据集
4. 通过返回的客户端数据进行联邦学习实验

## 6. 使用实例

### 6.1 基础用法：加载MNIST并划分为5个客户端

```python
from fed_dataloader import UniversalDataLoader

# 初始化数据加载器
loader = UniversalDataLoader(
    dataset_name='MNIST',
    num_client=5,          # 5个客户端
    num_local_class=2,     # 每个客户端分配2个类别
    root='../data',        # 数据存储目录
    seed=0,                # 随机种子
    in_channels=1,         # 输入通道数（MNIST为灰度图，1通道）
    size=32,               # 图像尺寸调整为32x32
    augmentation=True      # 启用数据增强
)

# 加载数据集
loader.load()

# 划分联邦数据集
trainset_config, testset = loader.divide()

# 查看数据集信息
print(f"客户端列表: {trainset_config['users']}")
print(f"总样本数: {trainset_config['num_samples']}")
print(f"测试集样本数: {len(testset)}")

# 获取第一个客户端的数据加载器
client_id = trainset_config['users'][0]
client_loader = trainset_config['user_data'][client_id].get_data_loader(batch_size=32)
```

### 6.2 高级用法：使用多种客户端策略

```python
# 初始化数据加载器，配置不同客户端策略
loader = UniversalDataLoader(
    dataset_name='CIFAR10',
    num_client=5,
    num_local_class=3,
    # 客户端策略：稀疏、特征噪声、类别倾斜、恶意、正常
    untrusted_strategies=[2.1, 3.5, 4.3, 0.5, 1.0],
    root='../data',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    preload_to_gpu=False,
    in_channels=3,
    size=32,
    augmentation=True
)

# 加载并划分数据集
loader.load()
trainset_config, testset = loader.divide()

# 验证前3个客户端的加载器
for i in range(3):
    client = trainset_config['user_data'][f"f_0000{i}"]
    client.verify_loader()  # 验证加载器并打印信息
```

### 6.3 自定义数据集用法

```python
# 加载自定义数据集
loader = UniversalDataLoader(
    dataset_name='my_dataset',  # 自定义数据集目录名
    num_client=4,
    num_local_class=5,
    root='../custom_data',      # 自定义数据根目录
    subset=('random', 10),      # 随机选择10个类别
    cut=0.8,                    # 只使用80%的数据
    size=(224, 224),            # 调整图像尺寸为224x224
    in_channels=3,              # RGB图像，3通道
    save=True,                  # 保存处理后的数据集
    seed=42
)

# 加载并划分数据集
loader.load()
trainset_config, testset = loader.divide()

# 计算数据集统计信息
mean, std = loader.calculate_statistics()
print(f"数据集均值: {mean}")
print(f"数据集标准差: {std}")
```

## 7. 注意事项

1. **数据集目录结构**：
   - 内置数据集会自动下载到指定的`root`目录
   - 自定义数据集需按照以下结构组织：
     ```
     root/
     └── dataset_name/
         ├── train/
         │   ├── class1/
         │   ├── class2/
         │   └── ...
         └── val/
             ├── class1/
             ├── class2/
             └── ...
     ```

2. **设备使用**：
   - 当`preload_to_gpu=True`时，数据会提前加载到GPU，可加速训练但消耗更多显存
   - 对于大规模数据集，建议设置`preload_to_gpu=False`，采用按需加载方式

3. **数据增强**：
   - 数据增强仅应用于训练集，测试集不进行增强处理
   - 增强操作在尺寸调整之后、归一化之前执行

4. **随机种子**：
   - 类内部已统一设置随机种子，确保实验结果可重复
   - 不同的`seed`值会导致不同的数据划分结果

5. **性能优化**：
   - 首次加载数据集时会进行预处理，可能耗时较长
   - 设置`save=True`可保存处理后的数据集，后续使用时可直接加载，提高效率

6. **类别分配**：
   - 当`num_local_class`大于总类别数时，会通过取模运算循环分配类别
   - 客户端之间的类别可能存在重叠

## 8. 扩展建议

1. 如需支持新的公开数据集，可扩展`_load_default_dataset`方法中的`default_datasets`字典
2. 如需添加新的数据增强方式，可修改`_get_augmentation_transform`方法
3. 如需自定义客户端数据分配策略，可重写`_assign_classes_to_clients`和`_create_clients`方法
4. 如需添加新的客户端策略，可扩展`parse_strategy`方法和`apply_strategy`方法

通过灵活配置`UniversalDataLoader`的各项参数，可以快速构建不同的数据分布场景，满足联邦学习算法的测试与验证需求。