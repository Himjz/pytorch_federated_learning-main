import itertools
import copy
import torch
import numpy as np


class ShapleyClient:
    def __init__(self, client_id, model, train_loader, criterion, optimizer, device):
        """
        初始化夏普利值客户端

        :param client_id: 客户端 ID
        :param model: 模型
        :param train_loader: 训练数据加载器
        :param criterion: 损失函数
        :param optimizer: 优化器
        :param device: 计算设备
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)

    def train(self):
        """
        客户端训练模型
        """
        self.model.train()
        total_loss = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def update(self, global_state_dict):
        """
        从服务端更新模型

        :param global_state_dict: 全局模型状态字典
        """
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def calculate_shapley_contribution(self, other_clients, evaluate_subset_func):
        """
        计算夏普利值贡献

        :param other_clients: 其他客户端列表
        :param evaluate_subset_func: 服务端提供的评估子集函数
        :return: 夏普利值
        """
        all_clients = [self] + other_clients
        n_clients = len(all_clients)
        shapley_value = 0
        # 计算所有可能的排列
        all_permutations = itertools.permutations(all_clients)
        num_permutations = np.math.factorial(n_clients)

        for permutation in all_permutations:
            subset = []
            prev_performance = 0

            for client in permutation:
                subset.append(client)
                # 调用服务端提供的评估函数
                performance = evaluate_subset_func([c.client_id for c in subset])
                marginal_contribution = performance - prev_performance
                if client == self:
                    shapley_value += marginal_contribution
                prev_performance = performance

        shapley_value /= num_permutations
        return shapley_value

    def _combine_models(self, clients):
        """
        合并多个客户端的模型

        :param clients: 客户端列表
        :return: 合并后的模型
        """
        combined_state_dict = copy.deepcopy(clients[0].model.state_dict())
        for key in combined_state_dict.keys():
            combined_state_dict[key] = torch.zeros_like(combined_state_dict[key])
            for client in clients:
                combined_state_dict[key] += client.model.state_dict()[key]
            combined_state_dict[key] /= len(clients)

        combined_model = copy.deepcopy(clients[0].model)
        combined_model.load_state_dict(combined_state_dict)
        return combined_model
