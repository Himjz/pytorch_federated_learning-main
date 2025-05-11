import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score, f1_score, precision_score
from torch.utils.data import DataLoader

from utils.fed_utils import assign_dataset, init_model


class FedServer(object):
    def __init__(self, client_list, dataset_id, model_name):
        """
        初始化联邦学习的服务器。
        :param client_list: 网络中连接的客户端列表
        :param dataset_id: 应用场景的数据集名称
        :param model_name: 应用场景的机器学习模型名称
        """
        # 初始化系统设置所需的字典和列表
        # 存储各客户端的模型状态字典
        self.client_state = {}
        # 存储各客户端的训练损失
        self.client_loss = {}
        # 存储各客户端的本地数据点数量
        self.client_n_data = {}
        # 存储选中参与本轮训练的客户端列表
        self.selected_clients = []
        # 测试时的批量大小
        self._batch_size = 200
        # 存储网络中连接的客户端列表
        self.client_list = client_list

        # 初始化测试数据集
        self.testset = None

        # 初始化服务器端联邦学习的超参数
        # 记录当前联邦学习的轮数
        self.round = 0
        # 记录本地数据点的总数
        self.n_data = 0
        # 记录应用场景的数据集名称
        self._dataset_id = dataset_id

        # 在 GPU 上进行测试
        # 指定使用的 GPU 编号
        gpu = 0
        # 根据 GPU 可用性选择设备
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        # 初始化全局机器学习模型
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)

    def load_testset(self, testset):
        """
        服务器加载测试数据集。
        :param testset: 用于测试的数据集。
        """
        self.testset = testset

    def state_dict(self):
        """
        服务器返回全局模型字典。
        :return: 全局模型字典
        """
        return self.model.state_dict()

    def test(self, default: bool = True):
        """
        服务器在测试数据集上测试模型。
        :param default: 若为 True，保持原输出；若为 False，依次返回准确率、召回率、F1 分数、损失、精确率
        """
        test_loader = DataLoader(self.testset, batch_size=self._batch_size, shuffle=True)
        self.model.to(self._device)
        accuracy_collector = 0
        loss_collector = 0
        all_preds = []
        all_labels = []

        for step, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                b_x = x.to(self._device)  # 将数据移到 GPU 上
                b_y = y.to(self._device)  # 将标签移到 GPU 上

                test_output = self.model(b_x)
                pred_y = torch.max(test_output, 1)[1].to(self._device).data.squeeze()

                # 累加预测正确的样本数量
                accuracy_collector += (pred_y == b_y).sum().item()

                # 计算损失
                loss = F.cross_entropy(test_output, b_y)
                loss_collector += loss.item()

                all_preds.extend(pred_y.cpu().numpy())
                all_labels.extend(b_y.cpu().numpy())

        accuracy = accuracy_collector / len(self.testset)
        avg_loss = loss_collector / len(test_loader)

        if default:
            return accuracy.cpu().numpy()
        else:
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')
            precision = precision_score(all_labels, all_preds, average='weighted')
            return accuracy, recall, f1, avg_loss, precision

    def select_clients(self, connection_ratio=1):
        """
        服务器选择一部分客户端。
        :param connection_ratio: 客户端的连接比例
        """
        # 选择一部分客户端
        # 初始化选中客户端列表
        self.selected_clients = []
        # 初始化本地数据点总数
        self.n_data = 0
        # 遍历所有客户端
        for client_id in self.client_list:
            # 依据连接比例进行二项分布采样
            b = np.random.binomial(np.ones(1).astype(int), connection_ratio)
            # 若采样结果为选中
            if b:
                # 将该客户端添加到选中客户端列表
                self.selected_clients.append(client_id)
                # 累加该客户端的本地数据点数量到总数
                self.n_data += self.client_n_data[client_id]

    def agg(self):
        """
        服务器聚合来自连接客户端的模型。
        :return: model_state: 聚合后更新的全局模型
        :return: avg_loss: 平均损失值
        :return: n_data: 本地数据点的数量
        """
        # 获取选中客户端的数量
        client_num = len(self.selected_clients)
        # 若没有选中客户端或本地数据点总数为 0
        if client_num == 0 or self.n_data == 0:
            # 直接返回当前全局模型、损失值 0 和本地数据点数量 0
            return self.model.state_dict(), 0, 0

        # 初始化一个用于聚合的模型
        model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        # 获取该模型的状态字典
        model_state = model.state_dict()
        # 初始化平均损失值
        avg_loss = 0

        # 聚合来自选中客户端的本地更新模型
        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                if i == 0:
                    model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[name] / self.n_data

            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        # 服务器加载聚合后的模型作为全局模型
        self.model.load_state_dict(model_state)
        self.round = self.round + 1
        n_data = self.n_data

        return model_state, avg_loss, n_data

    def rec(self, name, state_dict, n_data, loss):
        """
        服务器接收来自连接客户端 k 的本地更新。
        :param name: 客户端 k 的名称
        :param state_dict: 来自客户端 k 的模型字典
        :param n_data: 客户端 k 中本地数据点的数量
        :param loss: 客户端 k 中本地训练的损失
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss

    def flush(self):
        """
        清空服务器中的客户端信息
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
