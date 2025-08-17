import pickle
import socket
from typing import Dict, List, Any, Optional, Union

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from utils.fed_utils import init_model
from utils.models import *


class FedClient(object):
    def __init__(self, name, epoch, model_name, dataset_info: list | tuple,
                 local: bool = True, server_ip: str = '127.0.0.3', server_port: int = 9999,
                 enable_serialization: Optional[bool] = None):
        """
        初始化联邦学习中的客户端
        :param name: 客户端名称
        :param epoch: 客户端本地训练的轮数
        :param model_name: 客户端的本地模型名称
        :param dataset_info: 数据集信息，包含(num_class, image_dim, image_channel)
        :param local: 是否为本地模式
        :param server_ip: 服务器IP地址
        :param server_port: 服务器端口
        :param enable_serialization: 是否启用序列化，网络模式下此参数无效
        """
        # 客户端网络信息
        self.target_ip = server_ip
        self.port = server_port
        self.name = name
        self.socket: Optional[socket.socket] = None
        self.connected: bool = False

        # 训练参数
        self._epoch = epoch
        self._batch_size = 50
        self._lr = 0.001
        self._momentum = 0.9
        self.num_workers = 2
        self.loss_rec = []
        self.n_data = 0

        # 数据集
        self.trainset = None
        self.train_loader: Optional[DataLoader[Dataset]] = None

        # 模型相关
        self._num_class, self._image_dim, self._image_channel = dataset_info
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)

        # 计算模型参数数量
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])

        # 设备配置
        gpu = 0
        self._device = torch.device(
            f"cuda:{gpu}" if torch.cuda.is_available() and gpu != -1 else "cpu"
        )

        # 联邦学习模式配置
        self.local = local  # 是否为本地模式

        # 序列化开关逻辑
        # 本地模式: 默认关闭，可通过参数修改
        # 网络模式: 强制开启，忽略用户输入的参数
        if not self.local:  # 网络模式
            self.enable_serialization = True
            # 如果用户在网络模式下指定了序列化参数，给出警告
            if enable_serialization is not None and not enable_serialization:
                import warnings
                warnings.warn("网络模式下序列化始终开启，忽略 enable_serialization=False 的设置")
        else:  # 本地模式
            # 本地模式下使用用户指定的值，默认关闭
            self.enable_serialization = enable_serialization if enable_serialization is not None else False

        self.supported_algorithms: List[str] = ['fedavg', 'scaffold', 'fedprox']
        self.selected_algorithm: str = 'fedavg'

    def load_trainset(self, trainset):
        """客户端加载训练数据集"""
        self.trainset = trainset
        self.n_data = len(trainset)
        self.train_loader = DataLoader(
            trainset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def update(self, model_state_dict):
        """客户端从服务器更新模型"""
        self.model = init_model(model_name=self.model_name, num_class=self._num_class,
                                image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)

    def train(self):
        """客户端在本地数据集上训练模型"""
        if self.trainset is None:
            raise ValueError("未加载训练数据，请先调用load_trainset方法")

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

        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()

    # 网络通信功能
    def connect_to_server(self) -> bool:
        """连接到服务器（网络模式）"""
        if self.local:
            raise ValueError("本地模式下不需要连接服务器，请设置local=False")

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.target_ip, self.port))
            self.connected = True

            # 握手信息
            handshake_msg: Dict[str, Any] = {
                'type': 'handshake',
                'client_id': self.name,
                'supported_algorithms': self.supported_algorithms,
                'serialization_enabled': self.enable_serialization
            }

            # 应用编码和序列化
            encoded_data = self._encode(handshake_msg)
            send_data = self._serialize(encoded_data)
            self.socket.sendall(send_data)

            # 接收响应并反序列化、解码
            response_data = self.socket.recv(4096)
            deserialized_data = self._deserialize(response_data)
            response = self._decode(deserialized_data)

            if response.get('status') == 'success':
                print(f"客户端 {self.name} 成功连接到服务器")
                return True
            else:
                print(f"连接失败: {response.get('message', '未知错误')}")
                self.disconnect()
                return False

        except Exception as e:
            print(f"连接服务器错误: {e}")
            self.disconnect()
            return False

    def disconnect(self) -> None:
        """断开与服务器的连接"""
        if self.socket:
            self.socket.close()
        self.connected = False
        self.socket = None

    def get_global_model(self, round_num: Optional[int] = None) -> Optional[Dict[str, Tensor]]:
        """从服务器获取全局模型"""
        if self.local:
            return None  # 本地模式下通过其他方式获取

        if not self.connected and not self.connect_to_server():
            return None

        try:
            request: Dict[str, Any] = {
                'type': 'get_global_model',
                'client_id': self.name,
                'round': round_num
            }

            # 应用编码和序列化
            encoded_data = self._encode(request)
            send_data = self._serialize(encoded_data)
            self.socket.sendall(send_data)

            # 接收响应并反序列化、解码
            response_data = self.socket.recv(4096)
            deserialized_data = self._deserialize(response_data)
            response = self._decode(deserialized_data)

            if response.get('status') == 'success':
                return response.get('model_params')
            else:
                print(f"获取全局模型失败: {response.get('message')}")
                return None

        except Exception as e:
            print(f"获取全局模型错误: {e}")
            self.disconnect()
            return None

    def upload_local_model(self, local_results: tuple[Dict[str, Tensor], int, float]) -> bool:
        """向服务器上传本地模型"""
        if self.local:
            return True  # 本地模式下通过其他方式提交

        if not self.connected and not self.connect_to_server():
            return False

        try:
            model_params, data_size, loss = local_results
            request: Dict[str, Any] = {
                'type': 'upload_local_model',
                'client_id': self.name,
                'model_params': model_params,
                'loss': loss,
                'data_size': data_size
            }

            # 应用编码和序列化
            encoded_data = self._encode(request)
            send_data = self._serialize(encoded_data)
            self.socket.sendall(send_data)

            # 接收响应并反序列化、解码
            response_data = self.socket.recv(4096)
            deserialized_data = self._deserialize(response_data)
            response = self._decode(deserialized_data)

            return response.get('status') == 'success'

        except Exception as e:
            print(f"上传模型错误: {e}")
            self.disconnect()
            return False

    # 编码解码函数 - 预设为空实现，便于派生扩展
    def _encode(self, data: Any) -> Any:
        """
        数据编码处理（扩展接口）
        可在派生类中重写，用于数据加密、压缩或添加噪声等
        """
        return data  # 空实现，直接返回原始数据

    def _decode(self, data: Any) -> Any:
        """
        数据解码处理（扩展接口）
        可在派生类中重写，用于数据解密、解压或去除噪声等
        """
        return data  # 空实现，直接返回原始数据

    # 序列化工具方法
    def _serialize(self, data: Any) -> Union[bytes, Any]:
        """序列化数据"""
        if self.enable_serialization:
            return pickle.dumps(data)
        # 未启用序列化时直接返回数据本身
        return data

    def _deserialize(self, data: Any) -> Any:
        """反序列化数据"""
        if self.enable_serialization:
            return pickle.loads(data) if isinstance(data, bytes) else data
        # 未启用序列化时直接返回数据本身
        return data
