## 客户端数据的分布差异较大时，FedProx 通过近端项对本地模型更新进行约束，可以避免客户端模型偏离全局模型太远，从而提高模型在全局数据上的收敛速度和泛化能力。
import copy

from torch.utils.data import DataLoader

from fed_baselines.client_base import FedClient
from utils.models import *


class FedProxClient(FedClient):
    def __init__(self, name, epoch, model_name, dataset_info: list|tuple,
                 target_ip: str = '127.0.0.3', port: int = 9999,
                 local_ip: str = None, local_port: int = None,
                 return_packed: bool = False, local: bool = True
                 ):
        super().__init__(name, epoch, model_name, dataset_info,
                         target_ip, port,
                         local_ip, local_port,
                         return_packed, local)
        self.mu = 0.1

    def train(self):
        """
        客户端使用 FedProx 算法在本地数据集上训练模型
        :return: 本地更新后的模型、本地数据点数量、训练损失
        """
        # 创建数据加载器，用于批量加载本地训练数据
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        # 将模型移动到指定设备（如 GPU）上
        self.model.to(self._device)
        # 深拷贝当前模型的参数，作为全局模型参数的副本
        global_weights = copy.deepcopy(list(self.model.parameters()))

        # 使用随机梯度下降（SGD）优化器来更新模型参数
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # 注释掉的 Adam 优化器，可按需使用
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        # 使用交叉熵损失函数计算模型损失
        loss_func = nn.CrossEntropyLoss()

        # 用于收集每个 epoch 的损失值
        epoch_loss_collector = []

        # 注释掉的进度条，可用于显示训练进度
        # pbar = tqdm(range(self._epoch))
        # 开始训练多个 epoch
        for epoch in range(self._epoch):
            # 遍历数据加载器中的每个批次数据
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()

                    # fedprox
                    prox_term = 0.0
                    for p_i, param in enumerate(self.model.parameters()):
                        prox_term += (self.mu / 2) * torch.norm((param - global_weights[p_i])) ** 2
                    loss += prox_term
                    epoch_loss_collector.append(loss.item())

                    loss.backward()
                    optimizer.step()

        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()
