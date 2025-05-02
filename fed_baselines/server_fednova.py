from fed_baselines.server_base import FedServer
import copy


class FedNovaServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name):
        super().__init__(client_list, dataset_id, model_name)
        # 归一化系数
        self.client_coeff = {}
        # 归一化梯度
        self.client_norm_grad = {}

    def agg(self):
        """
        服务器使用 FedNova 算法聚合来自连接客户端的归一化模型
        :return: 聚合后更新的全局模型、平均损失值、本地数据点的数量
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        self.model.to(self._device)

        model_state = self.model.state_dict()
        nova_model_state = copy.deepcopy(model_state)
        avg_loss = 0
        coeff = 0.0
        for i, name in enumerate(self.selected_clients):
            # 累加归一化系数
            coeff = coeff + self.client_coeff[name] * self.client_n_data[name] / self.n_data
            for key in self.client_state[name]:
                if i == 0:
                    # 初始化聚合的归一化梯度
                    nova_model_state[key] = self.client_norm_grad[name][key] * self.client_n_data[name] / self.n_data
                else:
                    # 累加归一化梯度
                    nova_model_state[key] = nova_model_state[key] + self.client_norm_grad[name][key] * self.client_n_data[name] / self.n_data
            # 累加平均损失
            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        for key in model_state:
            # 根据 FedNova 算法更新全局模型
            model_state[key] -= coeff * nova_model_state[key]

        self.model.load_state_dict(model_state)

        self.round = self.round + 1

        return model_state, avg_loss, self.n_data

    def rec(self, name, state_dict, n_data, loss, coeff, norm_grad):
        """
        服务器接收来自连接客户端 k 的本地更新。
        :param name: 客户端 k 的名称
        :param state_dict: 来自客户端 k 的模型字典
        :param n_data: 客户端 k 中本地数据点的数量
        :param loss: 客户端 k 中本地训练的损失
        :param coeff: 归一化系数
        :param norm_grad: 归一化梯度
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}
        self.client_coeff[name] = -1
        self.client_norm_grad[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss
        self.client_coeff[name] = coeff
        self.client_norm_grad[name].update(norm_grad)


    def flush(self):
        """
        清空服务器中的客户端信息
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
        self.client_coeff = {}
        self.client_norm_grad = {}
