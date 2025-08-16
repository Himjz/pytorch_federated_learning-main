import copy

from torch.utils.data import DataLoader

from fed_baselines.client_base import FedClient
from utils.fed_utils import init_model
from utils.models import *


class ScaffoldClient(FedClient):
    def __init__(self, name, epoch, model_name, dataset_info: list|tuple,
                 target_ip: str = '127.0.0.3', port: int = 9999,
                 local_ip: str = None, local_port: int = None,
                 return_packed: bool = False, local: bool = True
                 ):
        super().__init__(name, epoch, model_name, dataset_info,
                         target_ip, port,
                         local_ip, local_port,
                         return_packed, local)
        # 服务器控制变量
        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        # 客户端控制变量
        self.ccv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)

    def update(self, model_state_dict, scv_state):
        """
        SCAFFOLD 客户端更新本地模型和服务器控制变量
        :param model_state_dict: 全局模型状态字典
        :param scv_state: 服务器控制变量状态字典
        """
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)
        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.scv.load_state_dict(scv_state)

    def train(self):
        """
        客户端使用 SCAFFOLD 算法在本地数据集上训练模型
        :return: 本地更新后的模型状态字典、本地数据点数量、训练损失、更新后的客户端控制变量状态字典
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        self.ccv.to(self._device)
        self.scv.to(self._device)
        # 深拷贝全局模型状态字典
        global_state_dict = copy.deepcopy(self.model.state_dict())
        # 获取服务器控制变量状态字典
        scv_state = self.scv.state_dict()
        # 获取客户端控制变量状态字典
        ccv_state = self.ccv.state_dict()
        # 计数器，记录训练步数
        cnt = 0

        # 使用随机梯度下降优化器
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # 注释掉的 Adam 优化器，可按需使用
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        # 交叉熵损失函数
        loss_func = nn.CrossEntropyLoss()

        # 用于收集每个 epoch 的损失值
        epoch_loss_collector = []

        # 训练过程
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    # 将数据移动到指定设备（如 GPU）
                    b_x = x.to(self._device)
                    b_y = y.to(self._device)

                with torch.enable_grad():
                    self.model.train()
                    # 前向传播
                    output = self.model(b_x)
                    # 计算损失
                    loss = loss_func(output, b_y.long())
                    # 清空优化器梯度
                    optimizer.zero_grad()

                    # 反向传播
                    loss.backward()
                    # 更新模型参数
                    optimizer.step()

                    # 获取更新后的模型状态字典
                    state_dict = self.model.state_dict()
                    for key in state_dict:
                        # 根据 SCAFFOLD 算法更新模型参数
                        state_dict[key] = state_dict[key] - self._lr * (scv_state[key] - ccv_state[key])
                    self.model.load_state_dict(state_dict)

                    cnt += 1
                    # 加载更新后的客户端控制变量状态
                    # 计算模型状态变化量
                    # 计算客户端控制变量状态变化量
                    # 根据 SCAFFOLD 算法更新客户端控制变量
                    # 获取当前模型状态字典
                    # 深拷贝客户端控制变量状态变化量
                    # 深拷贝更新后的客户端控制变量状态字典
                    # 深拷贝更新后的模型状态字典
                    epoch_loss_collector.append(loss.item())

        delta_model_state = copy.deepcopy(self.model.state_dict())

        new_ccv_state = copy.deepcopy(self.ccv.state_dict())
        delta_ccv_state = copy.deepcopy(new_ccv_state)
        state_dict = self.model.state_dict()
        for key in state_dict:
            new_ccv_state[key] = ccv_state[key] - scv_state[key] + (global_state_dict[key] - state_dict[key]) / (
                    cnt * self._lr)
            delta_ccv_state[key] = new_ccv_state[key] - ccv_state[key]
            delta_model_state[key] = state_dict[key] - global_state_dict[key]

        self.ccv.load_state_dict(new_ccv_state)

        return state_dict, self.n_data, loss.data.cpu().numpy(), new_ccv_state
