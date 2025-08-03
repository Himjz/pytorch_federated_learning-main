from fed_baselines import server_base
import itertools
import copy
import random

from preprocessing.fed_dataloader import DataSetInfo


class FedShapley(server_base.FedServer):
    def __init__(self, client_list, model_name, dataset_info: DataSetInfo, use_monte_carlo=False,
                 monte_carlo_samples=120):
        super().__init__(client_list, model_name, dataset_info)
        self.client_states = {}
        self.client_data_sizes = {}
        self.client_losses = {}
        self.use_monte_carlo = use_monte_carlo  # 开关控制是否使用蒙特卡洛方法
        self.monte_carlo_samples = monte_carlo_samples  # 蒙特卡洛采样数

    def agg(self):
        # 计算所有客户端的夏普利值
        shapley_values = self.calculate_shapley_values()

        # 初始化聚合后的模型参数
        aggregated_state_dict = copy.deepcopy(
            list(self.client_states.values())[0]) if self.client_states else self.model.state_dict()
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] = 0

        # 根据夏普利值加权聚合模型参数
        total_data = sum(self.client_data_sizes.values()) if self.client_data_sizes else 0
        avg_loss = 0
        for client_name, shapley_value in shapley_values.items():
            client_state = self.client_states[client_name]
            client_data = self.client_data_sizes[client_name]
            client_loss = self.client_losses[client_name]
            for key in aggregated_state_dict.keys():
                aggregated_state_dict[key] += shapley_value * client_state[key]
            avg_loss += client_loss * (client_data / total_data) if total_data > 0 else 0
        self.round = self.round + 1
        self.model.load_state_dict(aggregated_state_dict)
        return aggregated_state_dict, avg_loss, total_data

    def rec(self, name, state_dict, n_data, loss):
        # 记录客户端的模型参数、数据量和损失
        self.client_states[name] = state_dict
        self.client_data_sizes[name] = n_data
        self.client_losses[name] = loss
        # 确保客户端数据量被记录到 server_base 的 client_n_data 中
        self.client_n_data[name] = n_data

    def calculate_shapley_values(self):
        clients = list(self.client_states.keys())
        n_clients = len(clients)
        shapley_values = {client: 0 for client in clients}

        if self.use_monte_carlo:
            # 蒙特卡洛方法计算夏普利值
            for _ in range(self.monte_carlo_samples):
                permutation = random.sample(clients, n_clients)
                subset = set()
                for client in permutation:
                    marginal_contribution = self.evaluate_subset(subset.union({client})) - self.evaluate_subset(subset)
                    shapley_values[client] += marginal_contribution
                    subset.add(client)

            # 对每个客户端的夏普利值求平均
            for client in shapley_values.keys():
                shapley_values[client] /= self.monte_carlo_samples
        else:
            # 原始精确计算方法
            for subset_size in range(n_clients + 1):
                for subset in itertools.combinations(clients, subset_size):
                    subset = set(subset)
                    for client in clients:
                        if client not in subset:
                            # 计算包含客户端的子集和不包含客户端的子集的贡献差
                            marginal_contribution = self.evaluate_subset(subset.union({client})) - self.evaluate_subset(
                                subset)
                            # 计算夏普利值的系数
                            coefficient = (1 / n_clients) * (self.factorial(len(subset)) * self.factorial(
                                n_clients - len(subset) - 1) / self.factorial(n_clients - 1))
                            shapley_values[client] += coefficient * marginal_contribution

        # 归一化夏普利值
        total_shapley = sum(shapley_values.values())
        if total_shapley != 0:
            for client in shapley_values.keys():
                shapley_values[client] /= total_shapley
        return shapley_values

    def evaluate_subset(self, subset):
        # 模拟评估子集的贡献，这里简单使用损失的倒数作为贡献
        total_loss = 0
        for client in subset:
            total_loss += self.client_losses[client]
        return 1 / (total_loss + 1e-8)  # 避免除零错误

    def factorial(self, n):
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.factorial(n - 1)