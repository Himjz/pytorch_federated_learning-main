## 在客户端计算能力和数据量差异较大的场景中表现出色，能够自适应地调整客户端的更新权重，提高训练效率。
import copy

from torch.utils.data import DataLoader

from fed_baselines.client_base import FedClient
from preprocessing.fed_dataloader import DataSetInfo
from utils.models import *


class FedNovaClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, dataset_info: DataSetInfo):
        """
        初始化 FedNova 客户端。

        :param name: 客户端名称
        :param epoch: 训练轮数
        :param dataset_id: 数据集 ID
        :param model_name: 模型名称
        """
        super().__init__(name, epoch, dataset_id, model_name, dataset_info)
        self.rho = 0.9
        self._momentum = self.rho

    def train(self):
        """
        客户端使用 FedNova 算法在本地数据集上训练模型。

        :return: 本地更新后的模型状态字典、本地数据点数量、训练损失值（转换为 NumPy 数组）、归一化系数、归一化梯度
        """
        # 创建数据加载器，用于批量加载本地训练数据
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        # 将模型移动到指定设备（如 GPU）上进行训练
        self.model.to(self._device)
        # 深拷贝当前模型的状态字典，作为全局权重的初始值
        global_weights = copy.deepcopy(self.model.state_dict())

        # 使用随机梯度下降（SGD）优化器，设置学习率和动量
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # 注释掉的 Adam 优化器，可根据需要取消注释使用
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        # 定义交叉熵损失函数，用于分类任务
        loss_func = nn.CrossEntropyLoss()

        # 初始化本地更新步数计数器
        tau = 0
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                # 上下文管理器，禁用梯度计算，提高数据移动效率
                with torch.no_grad():
                    # 将输入数据张量移动到指定设备（如 GPU）
                    b_x = x.to(self._device)  # 位于 GPU 上的张量
                    # 将标签数据张量移动到指定设备（如 GPU）
                    b_y = y.to(self._device)  # 位于 GPU 上的张量

                # 上下文管理器，启用梯度计算，用于反向传播更新模型参数
                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    tau += 1

        coeff = (tau - self.rho * (1 - pow(self.rho, tau)) / (1 - self.rho)) / (1 - self.rho)

        state_dict = self.model.state_dict()
        norm_grad = copy.deepcopy(global_weights)
        for key in norm_grad:
            norm_grad[key] = torch.div(global_weights[key] - state_dict[key], coeff)

        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy(), coeff, norm_grad
