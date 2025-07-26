import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys

from pathlib import Path

# 解决模块导入路径问题
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from fed_baselines.client_base import FedClient

class FedDPClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name,
                 # 差分隐私参数
                 dp_epsilon=1.0, dp_alpha=0.1, delta=1e-5, sensitivity=1.0,
                 # 对抗训练参数
                 adv_epsilon=0.1, alpha0=0.01, lambda_decay=0.001,
                 adv_sigma=0.01, switch_round=10):
        super().__init__(name, epoch, dataset_id, model_name)
        # 差分隐私核心参数
        self.dp_epsilon = dp_epsilon  # 隐私预算ε（重命名避免冲突）
        self.dp_alpha = dp_alpha      # 预算调整系数
        self.delta = delta            # 松弛项δ
        self.sensitivity = sensitivity  # 敏感度△f
        # 对抗训练参数
        self.adv_epsilon = adv_epsilon  # 对抗扰动基础强度
        self.alpha0 = alpha0            # 初始步长（PGD/FGSM中的α）
        self.lambda_decay = lambda_decay  # 步长衰减率
        self.adv_sigma = adv_sigma      # 对抗样本中的高斯噪声标准差
        self.switch_round = switch_round  # 切换对抗方法的轮次阈值
        # 训练状态跟踪
        self.current_round = 0          # 全局训练轮次
        self.epoch_count = 0            # 累计训练 epoch 数

    def _compute_dp_noise_scale(self):
        """计算差分隐私噪声的标准差（原梯度噪声）"""
        return (np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity) / self.dp_epsilon

    def _compute_privacy_noise_scale(self):
        """计算最终输入样本的隐私保护噪声标准差"""
        # 可根据需求调整为与差分隐私参数关联的公式
        return self._compute_dp_noise_scale() * 0.5  # 示例：取梯度噪声的一半

    def _get_current_alpha(self):
        """动态计算当前轮次的对抗步长α_t"""
        return self.alpha0 * np.exp(-self.lambda_decay * self.current_round)

    def _generate_adversarial_samples(self, x, y):
        """生成对抗样本（支持FGSM/PGD动态切换）"""
        x_adv = x.clone().detach().requires_grad_(True)
        alpha_t = self._get_current_alpha()  # 动态步长
        loss_func = nn.CrossEntropyLoss()

        # 动态切换对抗方法：前期FGSM，后期PGD
        if self.current_round < self.switch_round:
            # FGSM方法（快速扰动）
            output = self.model(x_adv)
            loss = loss_func(output, y.long())
            # 只清除模型参数的梯度，不影响输入的梯度
            self.model.zero_grad()
            loss.backward()
            # 计算扰动：ε*符号梯度 + 高斯噪声
            grad_sign = x_adv.grad.sign()
            adv_perturb = self.adv_epsilon * grad_sign + torch.normal(
                0, self.adv_sigma, size=x_adv.shape, device=self._device
            )
            x_adv = x_adv + adv_perturb
        else:
            # PGD方法（多步迭代）
            for _ in range(5):  # 固定5步迭代，可动态调整
                output = self.model(x_adv)
                loss = loss_func(output, y.long())
                # 只清除模型参数的梯度，不影响输入的梯度
                self.model.zero_grad()
                # 清除上一步的梯度（如果存在）
                if x_adv.grad is not None:
                    x_adv.grad.zero_()
                loss.backward()
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + alpha_t * grad_sign
                # 施加L∞范数约束（自适应ε）
                x_adv = torch.clamp(x_adv, x - self.adv_epsilon, x + self.adv_epsilon)
                # 确保x_adv保持需要梯度的状态
                x_adv = x_adv.detach().requires_grad_(True)
            # 添加高斯噪声
            x_adv = x_adv + torch.normal(0, self.adv_sigma, size=x_adv.shape, device=self._device)

        # 确保像素值在合理范围（假设输入为图像，范围[0,1]）
        return torch.clamp(x_adv, 0.0, 1.0).detach()

    def _adjust_dp_epsilon(self):
        """动态调整差分隐私预算（原逻辑保留）"""
        self.dp_epsilon += self.dp_alpha
        self.current_round += 1

    def train(self):
        """整合对抗训练与双层噪声的训练流程"""
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)
        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = nn.CrossEntropyLoss()

        # 每轮训练前更新状态
        self._adjust_dp_epsilon()
        self.epoch_count += self._epoch

        # 训练主循环
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                # 数据预处理
                x = x.to(self._device)
                y = y.to(self._device)

                # 生成对抗样本并添加隐私噪声
                x_adv = self._generate_adversarial_samples(x, y)
                privacy_noise = torch.normal(
                    0, self._compute_privacy_noise_scale(), size=x_adv.shape, device=self._device
                )
                x_final = x_adv + privacy_noise  # 最终输入样本
                x_final = torch.clamp(x_final, 0.0, 1.0)  # 裁剪溢出值

                # 模型训练（使用带噪声的对抗样本）
                self.model.train()
                output = self.model(x_final)
                loss = loss_func(output, y.long())
                optimizer.zero_grad()
                loss.backward()

                # 对梯度添加差分隐私噪声
                dp_sigma = self._compute_dp_noise_scale()
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad += torch.normal(0, dp_sigma, size=param.grad.shape, device=self._device)

                optimizer.step()

        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()