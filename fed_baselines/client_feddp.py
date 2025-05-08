
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fed_baselines.client_base import FedClient


class FedDPClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, epsilon=1.0, alpha=0.1, delta=1e-5):
        super().__init__(name, epoch, dataset_id, model_name)
        # 差分隐私参数
        self.epsilon = epsilon          # 初始隐私预算
        self.alpha = alpha              # 隐私预算动态调整参数
        self.delta = delta              # 松弛项
        self.sensitivity = 1.0          # 敏感度（需根据具体任务调整）

    def _compute_noise_scale(self):
        """计算高斯噪声的标准差σ"""
        sigma = (np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon)
        return sigma

    def train(self):
        """重写训练方法，添加差分隐私噪声"""
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)
        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = nn.CrossEntropyLoss()

        # 动态调整隐私预算
        self.epsilon += self.alpha  # 简单线性增长，可根据需求调整策略

        # 训练过程
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)
                    b_y = y.to(self._device)

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()
                    loss.backward()

                    # 对梯度添加噪声
                    sigma = self._compute_noise_scale()
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.normal(0, sigma, size=param.grad.shape).to(self._device)
                            param.grad += noise

                    optimizer.step()

        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()