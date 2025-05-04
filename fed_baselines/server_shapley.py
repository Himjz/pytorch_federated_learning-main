# 从 fed_baselines 模块导入服务器基类，FedShapley 类将继承该基类
from fed_baselines import server_base
# 导入 itertools 模块，用于生成客户端的所有可能组合，在计算夏普利值时会用到
import itertools
# 导入 copy 模块，使用 deepcopy 方法来复制对象，避免引用传递带来的问题
import copy

class FedShapley(server_base.FedServer):
    """
    FedShapley 类继承自 server_base.FedServer，实现基于夏普利值的联邦学习聚合算法。
    夏普利值用于衡量每个客户端对整体模型的边际贡献，在模型聚合时会根据贡献进行加权。
    """
    def __init__(self, client_list, dataset_id, model_name):
        """
        初始化 FedShapley 类的实例。

        参数:
        client_list (list): 参与联邦学习的客户端名称列表。
        dataset_id (str): 所使用数据集的标识符，用于区分不同数据集。
        model_name (str): 所使用模型的名称，例如 "LeNet"。
        """
        # 调用父类 FedServer 的构造函数进行初始化
        super().__init__(client_list, dataset_id, model_name)
        # 字典，用于存储每个客户端的模型参数，键为客户端名称，值为模型参数状态字典
        self.client_states = {}
        # 字典，用于存储每个客户端的数据量，键为客户端名称，值为数据样本数量
        self.client_data_sizes = {}
        # 字典，用于存储每个客户端的训练损失，键为客户端名称，值为损失值
        self.client_losses = {}

    def agg(self):
        """
        根据客户端的夏普利值对模型参数进行加权聚合，并计算平均损失。

        返回:
        tuple: 包含聚合后的模型参数状态字典、平均损失和所有客户端数据总量的元组。
        """
        # 计算所有客户端的夏普利值
        shapley_values = self.calculate_shapley_values()

        # 初始化聚合后的模型参数
        # 如果有客户端的模型参数，则复制第一个客户端的参数结构；否则使用服务器模型的参数结构
        aggregated_state_dict = copy.deepcopy(list(self.client_states.values())[0]) if self.client_states else self.model.state_dict()
        # 将聚合后的模型参数初始化为 0
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] = 0

        # 根据夏普利值加权聚合模型参数
        # 计算所有客户端的数据总量
        total_data = sum(self.client_data_sizes.values()) if self.client_data_sizes else 0
        # 初始化平均损失
        avg_loss = 0
        # 遍历每个客户端及其对应的夏普利值
        for client_name, shapley_value in shapley_values.items():
            # 获取当前客户端的模型参数
            client_state = self.client_states[client_name]
            # 获取当前客户端的数据量
            client_data = self.client_data_sizes[client_name]
            # 获取当前客户端的训练损失
            client_loss = self.client_losses[client_name]
            # 根据夏普利值加权累加客户端的模型参数
            for key in aggregated_state_dict.keys():
                aggregated_state_dict[key] += shapley_value * client_state[key]
            # 计算加权平均损失，避免除零错误
            avg_loss += client_loss * (client_data / total_data) if total_data > 0 else 0

        return aggregated_state_dict, avg_loss, total_data

    def rec(self, name, state_dict, n_data, loss):
        """
        接收并记录客户端的模型参数、数据量和训练损失。

        参数:
        name (str): 客户端的名称。
        state_dict (dict): 客户端的模型参数状态字典。
        n_data (int): 客户端的数据样本数量。
        loss (float): 客户端的训练损失。
        """
        # 记录客户端的模型参数
        self.client_states[name] = state_dict
        # 记录客户端的数据量
        self.client_data_sizes[name] = n_data
        # 记录客户端的训练损失
        self.client_losses[name] = loss
        # 确保客户端数据量被记录到父类的 client_n_data 字典中
        self.client_n_data[name] = n_data

    def rec(self, name, state_dict, n_data, loss):
        """
        接收并记录客户端的模型参数、数据量和训练损失。

        参数:
        name (str): 客户端的名称。
        state_dict (dict): 客户端的模型参数状态字典。
        n_data (int): 客户端的数据样本数量。
        loss (float): 客户端的训练损失。
        """
        # 记录客户端的模型参数
        self.client_states[name] = state_dict
        # 记录客户端的数据量
        self.client_data_sizes[name] = n_data
        # 记录客户端的训练损失
        self.client_losses[name] = loss
        # 确保客户端数据量被记录到父类的 client_n_data 字典中
        self.client_n_data[name] = n_data
