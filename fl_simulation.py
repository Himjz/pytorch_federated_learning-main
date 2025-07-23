import torch
import numpy as np
from preprocessing.fed_dataloader import divide_data

# 配置参数
num_client = 10
num_local_class = 2
dataset_name = 'SelfDataSet'
i_seed = 235235

# 划分数据
trainset_config, testset = divide_data(
    num_client=num_client,
    num_local_class=num_local_class,
    dataset_name=dataset_name,
    i_seed=i_seed
)


# 辅助函数：统计并打印标签分布
def analyze_label_distribution(trainset_config, title):
    distribution = {}
    for client_id, client_data in trainset_config['user_data'].items():
        client_dist = {}
        for i in range(len(client_data)):
            _, label = client_data[i]
            label = int(label)
            if label not in client_dist:
                client_dist[label] = 1
            else:
                client_dist[label] += 1
        distribution[client_id] = client_dist

    print(f"\n{title}:")
    for client_id, dist in distribution.items():
        print(f"客户端 {client_id}: {dist}")

    # 计算全局分布
    global_dist = {}
    for client_dist in distribution.values():
        for label, count in client_dist.items():
            global_dist[label] = global_dist.get(label, 0) + count
    print(f"全局分布: {global_dist}")
    return distribution


# 分析原始数据分布
original_distribution = analyze_label_distribution(trainset_config, "原始标签分布")

# 获取最后一个客户端ID
last_client_id = list(trainset_config['user_data'].keys())[-1]

# 确定全局已知标签
global_labels = set()
for client_id, client_data in trainset_config['user_data'].items():
    for i in range(len(client_data)):
        _, label = client_data[i]
        global_labels.add(int(label))
global_labels = list(global_labels)
print(f"\n全局已知标签: {global_labels}")

# 获取最后一个客户端的原始标签
last_client_data = trainset_config['user_data'][last_client_id]
original_labels = np.array([int(last_client_data[i][1]) for i in range(len(last_client_data))])

# 在全局已知标签内进行随机标注
random_labels = np.random.choice(global_labels, size=len(last_client_data))


# 创建一个新的数据集替换原始Subset
class ModifiedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, original_indices, new_labels):
        self.original_dataset = original_dataset
        self.original_indices = original_indices
        self.new_labels = new_labels

    def __len__(self):
        return len(self.original_indices)

    def __getitem__(self, idx):
        original_idx = self.original_indices[idx]
        sample, _ = self.original_dataset[original_idx]
        return sample, self.new_labels[idx]


# 创建修改后的数据集
modified_dataset = ModifiedDataset(
    last_client_data.dataset,
    last_client_data.indices,
    random_labels
)

# 替换原始数据
trainset_config['user_data'][last_client_id] = modified_dataset

# 分析修改后的数据分布
modified_distribution = analyze_label_distribution(trainset_config, "随机标注后的标签分布")

# 计算标签错误率
error_mask = original_labels != random_labels
total_errors = np.sum(error_mask)
error_rate = total_errors / len(original_labels) * 100

print(f"\n最后一个客户端的标签错误率: {error_rate:.2f}% ({total_errors}/{len(original_labels)})")

# 分析每个标签的错误情况
label_error_counts = {label: 0 for label in global_labels}
label_total_counts = {label: 0 for label in global_labels}

for orig_label, rand_label, is_error in zip(original_labels, random_labels, error_mask):
    label_total_counts[orig_label] += 1
    if is_error:
        label_error_counts[orig_label] += 1

print("\n每个标签的错误情况:")
for label in sorted(global_labels):
    if label_total_counts[label] > 0:
        error_percentage = label_error_counts[label] / label_total_counts[label] * 100
        print(f"标签 {label}: {label_error_counts[label]}/{label_total_counts[label]} ({error_percentage:.2f}%)")