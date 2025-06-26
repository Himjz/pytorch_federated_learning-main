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
        self.client_state = {}
        self.client_loss = {}
        self.client_n_data = {}
        self.selected_clients = []
        self._batch_size = 200
        self.client_list = client_list

        # 初始化测试数据集
        self.testset = None

        # 初始化服务器端联邦学习的超参数
        self.round = 0
        self.n_data = 0
        self._dataset_id = dataset_id

        # 在 GPU 上进行测试
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        # 初始化全局机器学习模型
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)

    def load_testset(self, testset):
        """服务器加载测试数据集"""
        self.testset = testset

    def state_dict(self):
        """服务器返回全局模型字典"""
        return self.model.state_dict()

    def test(self, default: bool = True, confidence_threshold: float = 0.75,
             treat_invalid_as_negative: bool = True):
        """
        服务器在测试数据集上测试模型，支持置信度阈值过滤和无效预测负例处理
        :param default: 若为 True，返回准确率；若为 False，返回完整评估指标
        :param confidence_threshold: 置信度阈值，默认0.75
        :param treat_invalid_as_negative: 是否将无效预测（-1）视为负例，默认True
        :return: 评估指标（根据参数返回不同形式）
        """
        test_loader = DataLoader(self.testset, batch_size=self._batch_size, shuffle=True)
        self.model.to(self._device)
        accuracy_collector = 0
        loss_collector = 0
        all_preds = []
        all_labels = []

        for step, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                b_x = x.to(self._device)
                b_y = y.to(self._device)

                # 前向传播获取模型输出
                test_output = self.model(b_x)
                # 计算softmax得到类别概率分布
                probs = F.softmax(test_output, dim=1)
                # 获取原始预测类别（最大概率类别）
                pred_classes = torch.max(probs, 1)[1]

                # 检查所有类别概率是否均小于阈值
                all_below_threshold = torch.all(probs < confidence_threshold, dim=1)

                # 生成最终预测结果：无效预测标记为-1
                pred_y = torch.where(all_below_threshold,
                                     torch.full_like(pred_classes, -1, dtype=torch.long),
                                     pred_classes)

                # 仅计算可识别样本的准确率（原始逻辑）
                valid_mask = (pred_y != -1)
                valid_preds = pred_y[valid_mask]
                valid_labels = b_y[valid_mask]

                if valid_preds.numel() > 0:
                    accuracy_collector += (valid_preds == valid_labels).sum().item()

                # 计算损失（包含所有样本）
                loss = F.cross_entropy(test_output, b_y)
                loss_collector += loss.item()

                # 收集所有预测结果（包含-1）
                all_preds.extend(pred_y.cpu().numpy())
                all_labels.extend(b_y.cpu().numpy())

        accuracy = accuracy_collector / len(self.testset)
        avg_loss = loss_collector / len(test_loader)

        if default:
            return accuracy
        else:
            # 处理无效预测为负例的逻辑
            if treat_invalid_as_negative:
                # 创建负例标签（假设负例标签为0，多分类场景需调整）
                negative_class = 0
                all_preds_negative = [p if p != -1 else negative_class for p in all_preds]
                valid_preds = all_preds_negative
                valid_labels = all_labels
            else:
                # 不处理负例，仅过滤无效预测
                valid_preds = [p for p, l in zip(all_preds, all_labels) if p != -1]
                valid_labels = [l for p, l in zip(all_preds, all_labels) if p != -1]

            # 计算评估指标，添加zero_division=1参数
            recall = recall_score(valid_labels, valid_preds, average='weighted', zero_division=1)
            f1 = f1_score(valid_labels, valid_preds, average='weighted', zero_division=1)
            precision = precision_score(valid_labels, valid_preds, average='weighted', zero_division=1)

            return accuracy, recall, f1, avg_loss, precision

    def select_clients(self, connection_ratio=1):
        """服务器选择一部分客户端"""
        self.selected_clients = []
        self.n_data = 0
        for client_id in self.client_list:
            b = np.random.binomial(np.ones(1).astype(int), connection_ratio)
            if b:
                self.selected_clients.append(client_id)
                self.n_data += self.client_n_data[client_id]

    def agg(self):
        """服务器聚合来自连接客户端的模型"""
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        model_state = model.state_dict()
        avg_loss = 0

        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                if i == 0:
                    model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[
                        name] / self.n_data

            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        self.model.load_state_dict(model_state)
        self.round = self.round + 1
        n_data = self.n_data

        return model_state, avg_loss, n_data

    def rec(self, name, state_dict, n_data, loss):
        """服务器接收来自连接客户端的本地更新"""
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss

    def flush(self):
        """清空服务器中的客户端信息"""
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}