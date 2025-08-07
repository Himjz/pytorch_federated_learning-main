import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fed_baselines.client_base import FedClient
from preprocessing.fed_dataloader import DataSetInfo


class FedDPAdvClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, dataset_info: DataSetInfo,
                 # 差分隐私参数（梯度层）
                 dp_epsilon=10.0, dp_alpha=0.3, delta=1e-5,
                 sensitivity=0.3, clip_norm=1.0,  # 梯度裁剪阈值保持不变
                 # 自适应对抗训练参数（输入层）
                 adv_epsilon_init=0.05,  # 初始对抗扰动（后续会动态调整）
                 adv_lambda=0.002,  # 扰动衰减系数
                 adv_pgd_steps=3):  # PGD步数保持不变
        super().__init__(name, epoch, model_name, dataset_info)
        # 差分隐私参数
        self.dp_epsilon = dp_epsilon
        self.dp_alpha = dp_alpha
        self.delta = delta
        self.sensitivity = sensitivity
        self.clip_norm = clip_norm
        # 对抗训练参数
        self.adv_epsilon_init = adv_epsilon_init
        self.adv_lambda = adv_lambda
        self.adv_pgd_steps = adv_pgd_steps
        # 全局轮次计数器
        self.current_global_round = 0

    def _compute_noise_scale(self):
        """计算梯度层差分隐私噪声标准差，降低初始噪声强度"""
        decay = np.exp(-0.01 * self.current_global_round)
        # 核心调整：乘以0.7系数降低初始噪声，减少早期干扰
        return (np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity * decay * 0.7) / (self.dp_epsilon * 1.2)

    def _get_adaptive_params(self):
        """获取自适应对抗参数，弱化早期扰动并加快衰减"""
        # 核心调整：降低初始扰动幅度（0.05→0.03）并加快衰减（1.5倍系数）
        epsilon = (self.adv_epsilon_init * 0.6) * np.exp(-self.adv_lambda * 1.5 * self.current_global_round)
        epsilon = max(epsilon, 0.005)  # 核心调整：最低扰动从0.01降至0.005

        alpha = epsilon / self.adv_pgd_steps
        use_pgd = self.current_global_round >= 5  # PGD启动时机保持不变
        return epsilon, alpha, use_pgd

    def _generate_adv_samples(self, x, y):
        """生成更温和的对抗样本，减少噪声干扰"""
        epsilon, alpha, use_pgd = self._get_adaptive_params()
        x_adv = x.clone().detach().requires_grad_(True)
        loss_func = nn.CrossEntropyLoss()

        if use_pgd:
            for _ in range(self.adv_pgd_steps):
                output = self.model(x_adv)
                loss = loss_func(output, y)
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                x_adv = x_adv + 0.8 * alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
                x_adv = x_adv.detach().requires_grad_(True)
        else:
            output = self.model(x_adv)
            loss = loss_func(output, y)
            self.model.zero_grad()
            loss.backward()
            x_adv = x_adv + 0.5 * epsilon * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        grad_sigma = self._compute_noise_scale()
        input_sigma = grad_sigma * 0.1
        input_noise = torch.normal(0, input_sigma, size=x_adv.shape, device=self._device)
        x_adv = x_adv + input_noise
        return x_adv.detach()

    def train(self):
        train_loader = DataLoader(
            self.trainset,
            batch_size=self._batch_size,
            shuffle=True,
            pin_memory=(self._device.type == 'cpu')
        )
        self.model.to(self._device)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self._lr * 0.8,  # 保持学习率稳定性调整
            momentum=self._momentum
        )
        loss_func = nn.CrossEntropyLoss()

        self.dp_epsilon = min(15.0, self.dp_epsilon + self.dp_alpha)
        grad_sigma = self._compute_noise_scale()

        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                b_x = x.to(self._device)
                b_y = y.to(self._device)

                x_adv = self._generate_adv_samples(b_x, b_y)

                # 核心调整：分阶段控制对抗样本比例
                if self.current_global_round < 5:
                    # 前5轮：低对抗样本比例（1/4），优先学习原始数据
                    if step % 4 == 0:
                        train_x = x_adv
                    else:
                        train_x = b_x
                else:
                    # 5轮后：恢复原比例（1/2），保证对抗训练效果
                    if step % 2 == 0:
                        train_x = b_x
                    else:
                        train_x = x_adv

                self.model.train()
                output = self.model(train_x)
                loss = loss_func(output, b_y.long())
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

                # 核心调整：延迟梯度加噪至第5轮后（current_global_round > 4）
                if self.current_global_round > 4:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grad_noise = torch.normal(0, grad_sigma, size=param.grad.shape, device=self._device)
                            param.grad += grad_noise

                optimizer.step()

        self.current_global_round += 1
        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()