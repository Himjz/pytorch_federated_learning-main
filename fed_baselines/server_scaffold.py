from utils.fed_utils import init_model
from fed_baselines.server_base import FedServer
import copy


class ScaffoldServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name):
        super().__init__(client_list, dataset_id, model_name)
        # 服务器控制变量
        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        # 所有客户端控制变量的字典
        self.client_ccv_state = {}

    def agg(self):
        """
        服务器使用 SCAFFOLD 算法聚合来自连接客户端的归一化模型
        :return: 聚合后更新的全局模型、平均损失值、本地数据点数量、服务器控制变量
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        self.scv.to(self._device)
        self.model.to(self._device)
        model_state = self.model.state_dict()
        new_scv_state = copy.deepcopy(model_state)
        avg_loss = 0
        # print('number of selected clients in Cloud: ' + str(client_num))
        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:

                if i == 0:
                    model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                    new_scv_state[key] = self.client_ccv_state[name][key] * self.client_n_data[name] / self.n_data

                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[
                        name] / self.n_data
                    new_scv_state[key] = new_scv_state[key] + self.client_ccv_state[name][key] * self.client_n_data[
                        name] / self.n_data

            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        scv_state = self.scv.state_dict()

        self.model.load_state_dict(model_state)
        self.scv.load_state_dict(new_scv_state)
        self.round = self.round + 1

        return model_state, avg_loss, self.n_data, scv_state

    def rec(self, name, state_dict, n_data, loss, ccv_state):
        """
        服务器接收来自连接的客户端 k 的本地更新。
        :param name: 客户端 k 的名称
        :param state_dict: 来自客户端 k 的模型字典
        :param n_data: 客户端 k 中的本地数据点数量
        :param loss: 客户端 k 本地训练的损失值
        :param ccv_state: 归一化系数
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}
        self.client_ccv_state[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss
        self.client_ccv_state[name].update(ccv_state)


    def flush(self):
        """
        清除服务器中的客户端信息
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
        self.client_ccv_state = {}
