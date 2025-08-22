import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from fed_baselines.client_base import FedClient

class FedDPClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, dataset_info: list|tuple,
                 # 与FedDpAdv完全一致的DP参数，不可修改
                 dp_epsilon=15.0,
                 dp_alpha=0.1,
                 delta=1e-5,
                 sensitivity=0.3,
                 clip_norm=0.8):
        super().__init__(name, epoch, model_name, dataset_info)
        # DP核心参数（与FedDpAdv完全对齐）
        self.dp_epsilon = dp_epsilon
        self.dp_alpha = dp_alpha
        self.delta = delta
        self.sensitivity = sensitivity
        self.clip_norm = clip_norm
        self.current_global_round = 0  # 全局轮次计数器（与FedDpAdv一致）

    def _compute_noise_scale(self):
        """与FedDpAdv完全一致的噪声计算逻辑，不可修改"""
        if self.current_global_round < 20:
            decay = np.exp(-0.03 * self.current_global_round)
        else:
            decay = np.exp(-0.1 * (self.current_global_round - 20)) * np.exp(-0.03 * 20)
        return (np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity * decay * 0.2) / (self.dp_epsilon * 1.2)

    def train(self):
        # 数据加载（与FedDpAdv完全一致）
        train_loader = DataLoader(
            self.trainset,
            batch_size=self._batch_size,
            shuffle=True,
            pin_memory=(self._device.type == 'cpu')
        )
        self.model.to(self._device)
        # 优化器与学习率调度（与FedDpAdv完全一致）
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._lr if self._lr is not None else 0.001,
            weight_decay=1e-6  # 轻微权重衰减，避免过拟合
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.97)  # 与FedDpAdv一致
        loss_func = nn.CrossEntropyLoss()

        # DP逻辑（与FedDpAdv完全一致）
        self.dp_epsilon = min(35.0, self.dp_epsilon + self.dp_alpha)
        grad_sigma = self._compute_noise_scale()

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

                # 梯度裁剪（与FedDpAdv一致）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                # 梯度加噪（与FedDpAdv一致：10轮后加噪，30轮后噪声减半）
                if self.current_global_round > 10:
                    noise_scale = grad_sigma * 0.5 if self.current_global_round > 30 else grad_sigma
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grad_noise = torch.normal(0, noise_scale, size=param.grad.shape, device=self._device)
                            param.grad += grad_noise

                optimizer.step()
            scheduler.step()  # 每epoch更新学习率（与FedDpAdv一致）

        self.current_global_round += 1  # 轮次更新（与FedDpAdv一致）
        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()