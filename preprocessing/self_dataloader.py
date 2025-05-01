import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


def load_data(name, root='dt', download=True, save_pre_data=True):
    # 这里仅支持 SelfDataSet
    data_dict = ['SelfDataSet']
    assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == 'SelfDataSet':
        # 定义数据转换
        transform = transforms.Compose([
            transforms.Resize((803,803)),  # 调整图像大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 加载训练集
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'train'), transform=transform)
        # 加载测试集
        testset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'val'), transform=transform)

        # 将标签转换为张量
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)


    # 获取类别数量
    len_classes = len(trainset.classes)

    return trainset, testset, len_classes


def divide_data(num_client=1, num_local_class=10, dataset_name='SelfDataSet', i_seed=0):
    torch.manual_seed(i_seed)

    trainset, testset, len_classes = load_data(dataset_name, download=True, save_pre_data=False)

    num_classes = len_classes
    if num_local_class == -1:
        num_local_class = num_classes
    assert 0 < num_local_class <= num_classes, "number of local class should smaller than global number of class"

    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': 0}
    config_division = {}  # Count of the classes for division
    config_class = {}  # Configuration of class distribution in clients
    config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes

    for i in range(num_client):
        config_class['f_{0:05d}'.format(i)] = []
        for j in range(num_local_class):
            cls = (i + j) % num_classes
            if cls not in config_division:
                config_division[cls] = 1
                config_data[cls] = [0, []]
            else:
                config_division[cls] += 1
            config_class['f_{0:05d}'.format(i)].append(cls)

    # print(config_class)
    # print(config_division)

    for cls in config_division.keys():
        indexes = torch.nonzero(trainset.targets == cls)
        num_datapoint = indexes.shape[0]
        indexes = indexes[torch.randperm(num_datapoint)]
        num_partition = num_datapoint // config_division[cls]
        for i_partition in range(config_division[cls]):
            if i_partition == config_division[cls] - 1:
                config_data[cls][1].append(indexes[i_partition * num_partition:])
            else:
                config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])

    total_samples = 0
    for user in config_class.keys():
        user_data_indexes = torch.tensor([])
        for cls in config_class[user]:
            user_data_index = config_data[cls][1][config_data[cls][0]]
            user_data_indexes = torch.cat((user_data_indexes, user_data_index))
            config_data[cls][0] += 1
        user_data_indexes = user_data_indexes.squeeze().int().tolist()
        user_data = Subset(trainset, user_data_indexes)
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = user_data
        total_samples += len(user_data)

    trainset_config['num_samples'] = total_samples

    return trainset_config, testset


if __name__ == "__main__":
    data_dict = ['SelfDataSet']

    for name in data_dict:
        print(divide_data(num_client=5, num_local_class=2, dataset_name=name, i_seed=0))
