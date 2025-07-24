# 联邦学习数据预处理模块(Pro)使用说明

## 主要功能

该模块提供联邦学习场景下的数据集预处理功能，支持将集中式数据集划分为多个客户端本地数据集，并可对指定客户端应用多种不可信策略（如数据稀疏、特征噪声、类别倾斜、恶意标签篡改等），最终生成适用于联邦学习训练的客户端数据配置。

## 核心外函数

### 1. divide_data（核心函数）



```python
def divide_data(

   num_client=1,              # 客户端数量

   num_local_class=10,        # 每个客户端拥有的本地类别数

   dataset_name='emnist',     # 数据集名称（支持MNIST、CIFAR10等）

   i_seed=0,                  # 随机种子，保证结果可复现

   untrusted_strategies=None, # 不可信策略列表，每个元素对应一个客户端

   k=1,                       # 类别分配偏移参数

   print_report=True,         # 是否打印数据分布报告

   device=None,               # 计算设备（CPU/GPU）

   preload_to_gpu=False,      # 是否预加载数据到GPU

   load_path=None             # 自定义数据集路径（非默认数据集时使用）

):
    pass
```

**功能**：将原始数据集划分给多个客户端，支持应用不可信策略，返回训练集配置和测试集。

**返回值**：



*   `trainset_config`: 包含客户端列表、客户端数据和样本总数的字典

*   `testset`: 测试数据集

### 2. verify_loader



```python
def verify_loader(client_data, client_id, device):
    pass
```

**功能**：验证指定客户端的数据加载器是否正常工作，检查数据和标签的一致性。

### 3. 辅助函数



*   `load_data`: 加载指定数据集（支持默认数据集和自定义数据集）

*   `parse_strategy`: 解析策略代码，将数字代码转换为具体策略名称和参数

*   `apply_strategy`: 对客户端数据集应用指定的不可信策略

*   `print_distribution`: 打印客户端数据集的类别分布情况

*   `calculate_accuracy`: 计算策略应用后的标签准确率

## 不可信策略说明

通过`untrusted_strategies`列表指定，每个元素为一个浮点数，格式为`整数部分.小数部分`：



*   0.x: 恶意策略（标签篡改），小数部分表示标签准确率

*   2.x: 数据稀疏策略，小数部分表示保留样本比例

*   3.x: 特征噪声策略，小数部分表示添加噪声的样本比例

*   4.x: 类别倾斜策略，小数部分表示倾斜程度

*   1.0: 正常策略（无任何篡改）

## 使用示例

### 基础用法（无不可信策略）



```python
from preprocessing.fed_dataloader import divide_data, verify_loader
import torch
# 加载设备

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将MNIST数据集划分为5个客户端，每个客户端包含2个类别

trainset_config, testset = divide_data(

   num_client=5,

   num_local_class=2,

   dataset_name='MNIST',

   i_seed=0,

   device=device

)

# 验证第1个客户端的数据加载器

verify_loader(trainset_config['user_data'], 'f_00000', device)
```

### 含不可信策略的用法



```python
from preprocessing.fed_dataloader import divide_data, verify_loader
import torch
# 为5个客户端分别指定不同策略
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainset_config, testset = divide_data(

   num_client=5,

   num_local_class=2,

   dataset_name='MNIST',

   i_seed=0,

   # 策略列表：依次对应5个客户端

   untrusted_strategies=[2.1,  # 客户端0：数据稀疏（保留10%样本）

                         3.5,  # 客户端1：特征噪声（50%样本添加噪声）

                         4.3,  # 客户端2：类别倾斜（倾斜程度30%）

                         0.5,  # 客户端3：恶意策略（标签准确率50%）

                         1.0], # 客户端4：正常策略

   device=device,

   preload_to_gpu=False

)

# 验证所有客户端

for i in range(5):

   verify_loader(trainset_config['user_data'], f'f_0000{i}', device)
```

### 加载自定义数据集

```python
import torch

from preprocessing.fed_dataloader import divide_data

# 加载设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainset_config, testset = divide_data(

    num_client=3,

    num_local_class=5,

    dataset_name='custom',  # 自定义数据集

    untrusted_strategies=[2.3, 3.7, 1.0],

    device=device

)
```

## 策略说明



1.  **数据稀疏（2.x）**：参数 x 表示保留样本的比例（如 2.1 表示保留 10% 的样本）

2.  **特征噪声（3.x）**：参数 x 表示添加噪声的样本比例（如 3.5 表示 50% 的样本添加高斯噪声和亮度扰动）

3.  **类别倾斜（4.x）**：参数 x 表示倾斜程度，使客户端数据集中于某个主导类别

4.  **恶意策略（0.x）**：参数 x 表示标签准确率（如 0.5 表示 50% 的标签被恶意篡改）

5.  **正常策略（1.0）**：不做任何数据修改

## 注意事项



1.  客户端 ID 格式为`f_00000`、`f_00001`等，依次对应`untrusted_strategies`列表中的元素

2.  支持的默认数据集包括：MNIST、CIFAR10、FashionMNIST、CIFAR100

3.  自定义数据集需按照 ImageFolder 格式组织，包含 train 和 val 两个子目录

4.  使用 GPU 时，建议将`preload_to_gpu`设为 False，采用动态加载方式节省内存

5.  随机种子`i_seed`用于保证数据划分结果的可复现性

6.  通过调整`num_local_class`参数可以控制每个客户端拥有的类别数量，影响数据异构性
