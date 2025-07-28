import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fed_baselines.client_base import FedClient
from preprocessing.fed_dataloader import DataSetInfo


class FedDPClient(FedClient):
    def __init__(self, name, epoch, model_name, dataset_info: DataSetInfo,
                 # 差分隐私参数
                 dp_epsilon=1.0, dp_alpha=0.1, delta=1e-5, sensitivity=1.0,
                 # 对抗训练参数
                 adv_epsilon=0.1, alpha0=0.01, lambda_decay=0.001,
                 adv_sigma=0.01, switch_round=10,
                 # 新增参数：是否使用对抗训练
                 use_adv_training=True):
        super().__init__(name, epoch, model_name, dataset_info)
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
        # 新增属性：是否使用对抗训练
        self.use_adv_training = use_adv_training

    def load_testset(self, testset):
        """客户端加载测试数据集。"""
        self.testset = testset

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

        # 初始化损失变量
        loss = None

        # 训练主循环
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                # 数据预处理
                x = x.to(self._device)
                y = y.to(self._device)

                if self.use_adv_training:
                    # 生成对抗样本并添加隐私噪声
                    x_adv = self._generate_adversarial_samples(x, y)
                    privacy_noise = torch.normal(
                        0, self._compute_privacy_noise_scale(), size=x_adv.shape, device=self._device
                    )
                    x_final = x_adv + privacy_noise  # 最终输入样本
                    x_final = torch.clamp(x_final, 0.0, 1.0)  # 裁剪溢出值
                else:
                    # 不使用对抗训练，直接使用原始数据并添加隐私噪声
                    privacy_noise = torch.normal(
                        0, self._compute_privacy_noise_scale(), size=x.shape, device=self._device
                    )
                    x_final = x + privacy_noise  # 最终输入样本
                    x_final = torch.clamp(x_final, 0.0, 1.0)  # 裁剪溢出值

                # 模型训练（使用带噪声的样本）
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

        # 解决局部变量 'loss' 可能在赋值前引用的问题
        if loss is None:
            loss = torch.tensor(0.0, device=self._device)

        model_state = self.model.state_dict()
        # 进行成员推理攻击评估
        attack_accuracy = self.membership_inference_attack()
        return model_state, self.n_data, loss.data.cpu().numpy(), attack_accuracy

    def membership_inference_attack(self):
        """模拟成员推理攻击并评估性能"""
        if self.testset is None:
            raise ValueError("测试集未加载，请先调用 load_testset 方法加载测试集。")

        # 划分训练集和测试集的一部分作为攻击数据集
        train_size = int(len(self.trainset) * 0.8)
        train_subset, attack_in_train = torch.utils.data.random_split(self.trainset,
                                                                      [train_size, len(self.trainset) - train_size])
        test_size = int(len(self.testset) * 0.8)
        test_subset, attack_out_train = torch.utils.data.random_split(self.testset,
                                                                      [test_size, len(self.testset) - test_size])

        # 重新设置训练集
        self.trainset = train_subset

        # 准备攻击数据集
        attack_data = []
        attack_labels = []

        # 训练集中的样本标签为 1
        for x, y in attack_in_train:
            # 确保 x 是张量
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self._device).unsqueeze(0)
            output = self.model(x)
            confidence = torch.nn.functional.softmax(output, dim=1).max().item()
            attack_data.append(confidence)
            attack_labels.append(1)

        # 测试集中的样本标签为 0
        for x, y in attack_out_train:
            # 确保 x 是张量
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self._device).unsqueeze(0)
            output = self.model(x)
            confidence = torch.nn.functional.softmax(output, dim=1).max().item()
            attack_data.append(confidence)
            attack_labels.append(0)

        # 简单阈值判断：置信度高于阈值认为是训练集样本
        threshold = 0.9
        correct_predictions = 0
        for i in range(len(attack_data)):
            prediction = 1 if attack_data[i] >= threshold else 0
            if prediction == attack_labels[i]:
                correct_predictions += 1

        accuracy = correct_predictions / len(attack_data)
        return accuracy