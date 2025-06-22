## 客户端基类，实现客户端模型初始化和模型加载、更新、训练
from torch.utils.data import DataLoader

from utils.fed_utils import assign_dataset, init_model
from utils.models import *


class FedClient(object):
    def __init__(self, name, epoch, dataset_id, model_name):
        """
          初始化联邦学习中的客户端 k。
          :param name: 客户端 k 的名称
          :param epoch: 客户端 k 本地训练的轮数
          :param dataset_id: 客户端 k 的本地数据集
          :param model_name: 客户端 k 的本地模型
        """
        # 初始化本地客户端的元数据
        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name

        # 初始化本地客户端的参数
        self._epoch = epoch
        self._batch_size = 50
        self._lr = 0.001
        self._momentum = 0.9
        self.num_workers = 2
        self.loss_rec = []
        self.n_data = 0

        # 初始化本地训练和测试数据集
        self.trainset = None
        self.test_data = None

        # 初始化本地模型
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])

        # 在 GPU 上训练
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_trainset(self, trainset):
        """
        客户端加载训练数据集。
        :param trainset: 用于训练的数据集。
        """
        self.trainset = trainset
        self.n_data = len(trainset)

    def convert_to_malicious(self, ratio=1.0, random_seed=None):
        """
        将客户端转化为恶意客户端，在数据集有效标签内随机标注数据。
        :param ratio: 要进行随机标注的数据比例，范围为 0.0 - 1.0
        :param random_seed: 随机种子，用于复现随机标注的结果
        """
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("Ratio must be between 0.0 and 1.0")

        if random_seed is not None:
            np.random.seed(random_seed)

        num_data = len(self.trainset)
        num_to_modify = int(ratio * num_data)

        # 获取原始训练集的标签
        if hasattr(self.trainset, 'targets'):
            all_targets = self.trainset.targets
        elif hasattr(self.trainset, 'dataset') and hasattr(self.trainset.dataset, 'targets'):
            all_targets = self.trainset.dataset.targets
        else:
            raise AttributeError("Unable to access targets from the dataset.")

        # 获取数据集的有效标签
        valid_labels = np.unique(all_targets.numpy())

        # 随机选择要修改的样本索引
        indices_to_modify = np.random.choice(num_data, num_to_modify, replace=False)

        for idx in indices_to_modify:
            # 获取当前样本的真实标签
            if hasattr(self.trainset, 'targets'):
                true_label = self.trainset.targets[self.trainset.indices[idx]].item()
            elif hasattr(self.trainset, 'dataset') and hasattr(self.trainset.dataset, 'targets'):
                true_label = self.trainset.dataset.targets[self.trainset.indices[idx]].item()
            else:
                raise AttributeError("Unable to access targets from the dataset.")

            # 随机选择一个不同的有效标签
            new_label = np.random.choice(np.delete(valid_labels, np.where(valid_labels == true_label)))

            # 修改标签
            if hasattr(self.trainset, 'targets'):
                self.trainset.targets[self.trainset.indices[idx]] = torch.tensor(new_label)
            elif hasattr(self.trainset, 'dataset') and hasattr(self.trainset.dataset, 'targets'):
                self.trainset.dataset.targets[self.trainset.indices[idx]] = torch.tensor(new_label)
            else:
                raise AttributeError("Unable to modify targets in the dataset.")

    def update(self, model_state_dict):
        """
        客户端从服务器更新模型。
        :param model_state_dict: 全局模型。
        """
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)

    def train(self):
        """
        客户端在本地数据集上训练模型
        :return: 本地更新后的模型、本地数据点数量、训练损失
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss()

        # 训练过程
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # GPU 上的张量
                    b_y = y.to(self._device)  # GPU 上的张量

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()
