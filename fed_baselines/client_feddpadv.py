import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from fed_baselines.client_base import FedClient

class FedDPAdvClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, dataset_info: list|tuple,
                 # 1. 完全对齐修改后的FedDp的DP参数
                 dp_epsilon=15.0,
                 dp_alpha=0.1,
                 delta=1e-5,
                 sensitivity=0.4,  # 与FedDp一致：0.4
                 clip_norm=1.0,     # 与FedDp一致：1.0
                 # 2. 优化对抗训练参数（关键调整点）
                 adv_epsilon_init=0.03,  # 原0.025，增强初始扰动
                 adv_lambda=0.001,       # 原0.002，减缓扰动衰减
                 adv_pgd_steps=6):       # 原5，生成更精细对抗样本
        super().__init__(name, epoch, model_name, dataset_info)
        # 2.1 DP参数（与调整后的FedDp完全一致，不可修改）
        self.dp_epsilon = dp_epsilon
        self.dp_alpha = dp_alpha
        self.delta = delta
        self.sensitivity = sensitivity
        self.clip_norm = clip_norm
        # 2.2 优化后的对抗训练参数
        self.adv_epsilon_init = adv_epsilon_init
        self.adv_lambda = adv_lambda
        self.adv_pgd_steps = adv_pgd_steps
        self.current_global_round = 0  # 与FedDp一致

    def _compute_noise_scale(self):
        """3. 与调整后的FedDp完全一致，不可修改"""
        if self.current_global_round < 20:
            decay = np.exp(-0.03 * self.current_global_round)
        else:
            decay = np.exp(-0.1 * (self.current_global_round - 20)) * np.exp(-0.03 * 20)
        return (np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity * decay * 0.2) / (self.dp_epsilon * 1.2)

    def _get_adaptive_params(self):
        """4. 优化对抗扰动衰减：更慢衰减，保持后期扰动强度"""
        if self.current_global_round < 30:
            # 衰减系数从0.5→0.3，基数0.4不变，扰动下降更平缓
            epsilon = (self.adv_epsilon_init * 0.4) * np.exp(-self.adv_lambda * 0.3 * self.current_global_round)
            epsilon = max(epsilon, 0.01)  # 原0.008，提升最小扰动
        else:
            # 30轮后衰减系数从0.3→0.1，进一步减缓衰减
            epsilon = (self.adv_epsilon_init * 0.4) * np.exp(-self.adv_lambda * 0.3 * 30) * np.exp(
                -self.adv_lambda * 0.1 * (self.current_global_round - 30))
            epsilon = max(epsilon, 0.008)  # 稳定后期扰动
        alpha = epsilon / self.adv_pgd_steps  # 步长与新增的6步PGD匹配
        use_pgd = self.current_global_round >= 6  # 保持提前启动PGD
        return epsilon, alpha, use_pgd

    def _generate_adv_samples(self, x, y):
        """5. 优化对抗样本生成：更高质量，更少失真"""
        epsilon, alpha, use_pgd = self._get_adaptive_params()
        x_adv = x.clone().detach().requires_grad_(True)
        loss_func = nn.CrossEntropyLoss()

        if use_pgd:
            # 步长系数从0.95→0.97（30轮前）、0.98→0.99（30轮后），减少过度扰动
            step_coeff = 0.97 if self.current_global_round < 30 else 0.99
            for _ in range(self.adv_pgd_steps):  # 6步PGD
                output = self.model(x_adv)
                loss = loss_func(output, y)
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                # 平缓更新扰动，避免样本失真
                x_adv = x_adv + step_coeff * alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)  # 保留像素约束
                x_adv = x_adv.detach().requires_grad_(True)
                assert x_adv.shape == x.shape, f"对抗样本形状不匹配！原始: {x.shape}, 生成: {x_adv.shape}"
            return x_adv
        else:
            # FGSM阶段：扰动系数从0.7→0.8，增强初始对抗效果
            output = self.model(x_adv)
            loss = loss_func(output, y)
            self.model.zero_grad()
            loss.backward()
            x_adv = x_adv + 0.8 * epsilon * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # 6. 优化输入噪声：减少干扰，提升样本质量
        grad_sigma = self._compute_noise_scale()
        # 输入噪声比例：前20轮0.008→0.006，20轮后0.005→0.003，减少噪声对样本的影响
        input_sigma_ratio = 0.006 if self.current_global_round < 20 else 0.003
        input_sigma = grad_sigma * input_sigma_ratio
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
        # 7. 优化器与学习率调度：与调整后的FedDp完全一致
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._lr if self._lr is not None else 0.001,
            weight_decay=1e-6
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.97)
        loss_func = nn.CrossEntropyLoss()

        # 8. DP逻辑：与调整后的FedDp完全一致
        self.dp_epsilon = min(35.0, self.dp_epsilon + self.dp_alpha)
        grad_sigma = self._compute_noise_scale()

        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                b_x = x.to(self._device)
                b_y = y.to(self._device)
                # 9. 生成优化后的对抗样本
                x_adv = self._generate_adv_samples(b_x, b_y)
                # 10. 优化对抗样本混合比例：更多对抗样本参与训练
                if self.current_global_round < 10:
                    mix_ratio = 0.1    # 原0.125，早期少量适应
                elif self.current_global_round < 30:
                    mix_ratio = 0.2    # 原0.167，中期增加比例
                else:
                    mix_ratio = 0.3    # 原0.2，后期更多对抗样本提升鲁棒性
                # 随机混合（每批都有，避免波动）
                mask = torch.rand(b_x.shape[0], device=self._device) < mix_ratio
                mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                train_x = torch.where(mask, x_adv, b_x)

                # 11. 模型训练：仅输入为混合样本，其余与FedDp一致
                self.model.train()
                output = self.model(train_x)
                loss = loss_func(output, b_y.long())
                optimizer.zero_grad()
                loss.backward()

                # 12. DP梯度处理：与调整后的FedDp完全一致
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                if self.current_global_round > 10:
                    noise_scale = grad_sigma * 0.5 if self.current_global_round > 30 else grad_sigma
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grad_noise = torch.normal(0, noise_scale, size=param.grad.shape, device=self._device)
                            param.grad += grad_noise
                optimizer.step()
            scheduler.step()
        self.current_global_round += 1
        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()