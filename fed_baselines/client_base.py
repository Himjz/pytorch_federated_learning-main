from torch.utils.data import DataLoader
import torch
import numpy as np
import socket
import pickle
import struct

from utils.fed_utils import init_model
from utils.models import *


class FedClient(object):
    def __init__(self, name, epoch, model_name, dataset_info: list | tuple,
                 target_ip: str = '127.0.0.3', target_port: int = 9999,
                 local_ip: str = None, local_port: int = None,
                 return_packed: bool = False, local: bool = True):
        """
          初始化联邦学习中的客户端 k。
          :param name: 客户端 k 的名称
          :param epoch: 客户端 k 本地训练的轮数
          :param model_name: 客户端 k 的本地模型
          :param target_ip: 目标服务器IP地址（网络模式有效）
          :param target_port: 目标服务器端口（网络模式有效）
          :param local_ip: 客户端本地IP地址（网络模式有效）
          :param local_port: 客户端本地端口（网络模式有效）
          :param return_packed: 若为True，train()方法会返回序列化打包的训练结果
          :param local: 若为True，启用本地模式（隐藏网络传输功能），默认为True
        """
        # 核心参数初始化
        self.name = name
        self._epoch = epoch
        self.model_name = model_name
        self.return_packed = return_packed
        self.local = local  # 本地模式开关

        # 数据集信息初始化
        self._num_class, self._image_dim, self._image_channel = dataset_info

        # 训练参数初始化
        self._batch_size = 50
        self._lr = 0.001
        self._momentum = 0.9
        self.num_workers = 2
        self.loss_rec = []
        self.n_data = 0

        # 训练结果存储
        self.train_result = None
        self.packed_result = None  # 序列化结果在本地模式也可使用

        # 数据集存储
        self.trainset = None
        self.test_data = None

        # 模型初始化
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])

        # 设备配置
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        # 网络相关参数（仅在非本地模式下有效）
        if not self.local:
            self.target_ip = target_ip
            self.target_port = target_port
            self.client_ip = local_ip if local_ip else self._get_local_ip()
            self.client_port = local_port
            self.is_connected = False
        else:
            # 本地模式下隐藏网络通信参数
            self.target_ip = None
            self.target_port = None
            self.client_ip = None
            self.client_port = None
            self.is_connected = False

    def _get_local_ip(self):
        """获取本地IP地址（仅网络模式使用）"""
        if self.local:
            raise RuntimeError("本地模式下不支持网络操作")

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def load_trainset(self, trainset):
        """加载训练数据集"""
        self.trainset = trainset
        self.n_data = len(trainset)

    def update(self, model_state_dict):
        """从服务器更新模型"""
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)

    def train(self):
        """本地训练模型"""
        if not self.trainset:
            raise ValueError("请先加载训练数据集")

        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = nn.CrossEntropyLoss()

        # 训练过程
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)
                    b_y = y.to(self._device)

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # 结果处理
        model_state = self.model.state_dict()
        n_data = self.n_data
        final_loss = loss.data.cpu().numpy()

        self.train_result = {
            'model_state': model_state,
            'n_data': n_data,
            'loss': final_loss,
            'client_name': self.name
        }

        # 只要return_packed为True就打包结果（本地模式也可使用）
        if self.return_packed:
            self.packed_result = self._pack_result()

        # 返回结果
        if self.return_packed:
            return model_state, n_data, final_loss, self.packed_result
        else:
            return model_state, n_data, final_loss

    def _pack_result(self):
        """序列化训练结果（本地模式和网络模式均可使用）"""
        if not self.train_result:
            raise ValueError("没有可打包的训练结果，请先执行train()")

        try:
            return pickle.dumps(self.train_result)
        except Exception as e:
            raise RuntimeError(f"打包训练结果失败: {str(e)}")

    def upload(self, target_ip=None, port=None):
        """上传训练结果（仅网络模式使用）"""
        if self.local:
            raise RuntimeError("本地模式下不支持上传操作")

        if not self.packed_result:
            raise RuntimeError("没有可上传的训练结果，请先执行训练并设置return_packed=True")

        server_ip = target_ip or self.target_ip
        server_port = port or self.target_port

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if self.client_port:
                    s.bind((self.client_ip, self.client_port))
                s.connect((server_ip, server_port))

                data_len = struct.pack('>I', len(self.packed_result))
                s.sendall(data_len)
                s.sendall(self.packed_result)

            print(f"客户端 {self.name} 成功上传训练结果到 {server_ip}:{server_port}")
            return True

        except Exception as e:
            print(f"上传训练结果失败: {str(e)}")
            return False

    def connect(self, target_ip=None, port=None):
        """与服务器建立连接（仅网络模式使用）"""
        if self.local:
            raise RuntimeError("本地模式下不支持连接操作")

        server_ip = target_ip or self.target_ip
        server_port = port or self.target_port

        try:
            connect_info = {
                'client_name': self.name,
                'client_ip': self.client_ip,
                'client_port': self.client_port,
                'timestamp': torch.datetime.datetime.now().isoformat()
            }

            data = pickle.dumps(connect_info)

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if self.client_port:
                    s.bind((self.client_ip, self.client_port))
                s.connect((server_ip, server_port))

                data_len = struct.pack('>I', len(data))
                s.sendall(data_len)
                s.sendall(data)

                response = s.recv(1024)
                if response == b'ACK':
                    self.is_connected = True
                    print(f"客户端 {self.name} 成功连接到服务器 {server_ip}:{server_port}")
                    return True
                else:
                    print(f"服务器 {server_ip}:{server_port} 拒绝了连接请求")
                    return False

        except Exception as e:
            print(f"建立连接失败: {str(e)}")
            self.is_connected = False
            return False

    # 隐藏网络相关方法（在本地模式下尝试调用时会抛出异常）
    def __getattribute__(self, name):
        """属性访问控制，本地模式下禁用网络方法"""
        if name in ['upload', 'connect', '_get_local_ip'] and object.__getattribute__(self, 'local'):
            def disabled_method(*args, **kwargs):
                raise RuntimeError(f"本地模式下禁用 {name} 方法")

            return disabled_method
        return object.__getattribute__(self, name)
