import pickle
import socket
from typing import Dict, List, Any, Optional, Union

import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class FedClient:
    def __init__(self,
                 client_id: str,
                 model_name: str,
                 dataset_name: str,
                 device: str = 'cpu',
                 local: bool = True,
                 server_ip: str = 'localhost',
                 server_port: int = 5000,
                 enable_serialization: Optional[bool] = None) -> None:
        # 客户端基础属性
        self.client_id: str = client_id
        self.model_name: str = model_name
        self.dataset_name: str = dataset_name
        self.device: str = device
        self.local_model: nn.Module = self._init_local_model()
        self.train_loader: Optional[DataLoader[Dataset]] = None
        self.val_loader: Optional[DataLoader[Dataset]] = None
        self.epochs: int = 3
        self.batch_size: int = 32
        self.lr: float = 0.01

        # 网络相关属性
        self.local: bool = local  # 是否为本地模式
        self.server_ip: str = server_ip
        self.server_port: int = server_port
        self.socket: Optional[socket.socket] = None
        self.connected: bool = False
        self.supported_algorithms: List[str] = ['fedavg', 'scaffold', 'fedprox']
        self.selected_algorithm: str = 'fedavg'

        # 序列化控制参数
        if enable_serialization is None:
            self.enable_serialization: bool = not self.local
        else:
            # 网络模式下忽略用户设置，强制开启
            self.enable_serialization = enable_serialization if self.local else True

    def _init_local_model(self) -> nn.Module:
        """初始化本地模型"""
        if self.model_name == 'cnn' and self.dataset_name == 'mnist':
            model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(13 * 13 * 32, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        else:
            model = nn.Linear(20, 10)

        return model.to(self.device)

    def load_data(self, train_data: Dataset, val_data: Optional[Dataset] = None) -> 'FedClient':
        """加载本地数据集"""
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        if val_data:
            self.val_loader = DataLoader(val_data, batch_size=self.batch_size)
        return self

    def set_train_params(self,
                         epochs: Optional[int] = None,
                         batch_size: Optional[int] = None,
                         lr: Optional[float] = None) -> 'FedClient':
        """设置训练参数"""
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if lr is not None:
            self.lr = lr
        return self

    def train(self, global_model_params: Optional[Union[Dict[str, Tensor], bytes]] = None) -> Dict[
        str, Union[Dict[str, Tensor], bytes, float, int]]:
        """本地训练模型"""
        # 如果提供了全局模型参数，先加载
        if global_model_params is not None:
            # 根据序列化设置处理参数
            processed_params: Dict[str, Tensor]
            if self.enable_serialization:
                if isinstance(global_model_params, bytes):
                    deserialized = self._deserialize(global_model_params)
                    processed_params = self._decode(deserialized)
                else:
                    processed_params = self._decode(global_model_params)  # type: ignore
            else:
                processed_params = global_model_params if isinstance(global_model_params, dict) else {}  # type: ignore

            self.local_model.load_state_dict(processed_params)

        self.local_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.lr)

        total_loss: float = 0.0
        for epoch in range(self.epochs):
            epoch_loss: float = 0.0
            if self.train_loader is None:
                raise ValueError("未加载训练数据，请先调用load_data方法")

            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.local_model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss: float = epoch_loss / len(self.train_loader)
            total_loss += avg_epoch_loss

        # 返回训练后的参数、平均损失和数据量
        local_params: Dict[str, Tensor] = self.local_model.state_dict()
        processed_output: Union[Dict[str, Tensor], bytes] = local_params

        if self.enable_serialization:
            encoded = self._encode(local_params)
            processed_output = self._serialize(encoded)

        return {
            'params': processed_output,
            'loss': total_loss / self.epochs,
            'data_size': len(self.train_loader.dataset)
        }

    def connect_to_server(self) -> bool:
        """连接到服务器（网络模式）"""
        if self.local:
            raise ValueError("本地模式下不需要连接服务器，请设置local=False")

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self.connected = True

            # 握手
            handshake_msg: Dict[str, Any] = {
                'type': 'handshake',
                'client_id': self.client_id,
                'supported_algorithms': self.supported_algorithms
            }

            send_data: Union[bytes, Dict[str, Any]]
            if self.enable_serialization:
                send_data = self._serialize(handshake_msg)
            else:
                send_data = handshake_msg

            self.socket.sendall(send_data if isinstance(send_data, bytes) else str(send_data).encode())  # type: ignore

            # 接收响应
            response_data: bytes = self.socket.recv(4096)
            response: Dict[str, Any]

            if self.enable_serialization:
                response = self._deserialize(response_data)
            else:
                response = eval(response_data.decode())  # 仅用于本地非序列化模式

            if response.get('status') == 'success':
                self.client_id = response.get('client_id', self.client_id)
                print(f"客户端 {self.client_id} 成功连接到服务器")
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

    def get_global_model(self, round_num: Optional[int] = None) -> Optional[Union[Dict[str, Tensor], bytes]]:
        """从服务器获取全局模型"""
        if self.local:
            return None  # 本地模式下通过其他方式获取全局模型

        if not self.connected and not self.connect_to_server():
            return None

        try:
            request: Dict[str, Any] = {
                'type': 'get_global_model',
                'client_id': self.client_id,
                'round': round_num
            }

            send_data: Union[bytes, Dict[str, Any]]
            if self.enable_serialization:
                send_data = self._serialize(request)
            else:
                send_data = request

            self.socket.sendall(send_data if isinstance(send_data, bytes) else str(send_data).encode())  # type: ignore

            response_data: bytes = self.socket.recv(4096)
            response: Dict[str, Any]

            if self.enable_serialization:
                response = self._deserialize(response_data)
            else:
                response = eval(response_data.decode())  # 仅用于本地非序列化模式

            if response.get('status') == 'success':
                return response.get('model_params')
            else:
                print(f"获取全局模型失败: {response.get('message', '未知错误')}")
                return None

        except Exception as e:
            print(f"获取全局模型错误: {e}")
            self.disconnect()
            return None

    def upload_local_model(self, local_results: Dict[str, Union[Dict[str, Tensor], bytes, float, int]]) -> bool:
        """向服务器上传本地模型"""
        if self.local:
            return True  # 本地模式下通过其他方式提交模型

        if not self.connected and not self.connect_to_server():
            return False

        try:
            request: Dict[str, Any] = {
                'type': 'upload_local_model',
                'client_id': self.client_id,
                'model_params': local_results['params'],
                'loss': local_results['loss'],
                'data_size': local_results['data_size']
            }

            send_data: Union[bytes, Dict[str, Any]]
            if self.enable_serialization:
                send_data = self._serialize(request)
            else:
                send_data = request

            self.socket.sendall(send_data if isinstance(send_data, bytes) else str(send_data).encode())  # type: ignore

            response_data: bytes = self.socket.recv(4096)
            response: Dict[str, Any]

            if self.enable_serialization:
                response = self._deserialize(response_data)
            else:
                response = eval(response_data.decode())  # 仅用于本地非序列化模式

            if response.get('status') == 'success':
                return True
            else:
                print(f"上传本地模型失败: {response.get('message', '未知错误')}")
                return False

        except Exception as e:
            print(f"上传本地模型错误: {e}")
            self.disconnect()
            return False

    def run_federated_round(self, round_num: int) -> bool:
        """执行一轮联邦学习"""
        print(f"客户端 {self.client_id} 开始第 {round_num} 轮联邦学习")

        # 获取全局模型
        global_model = self.get_global_model() if not self.local else None

        # 本地训练
        local_results = self.train(global_model)

        # 上传本地模型
        success = self.upload_local_model(local_results) if not self.local else True

        return success

    # 序列化和编码工具方法
    def _serialize(self, data: Any) -> bytes:
        """序列化数据（将Python对象转换为可传输的字节流）"""
        if not self.enable_serialization:
            return str(data).encode() if not isinstance(data, bytes) else data
        return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> Any:
        """反序列化数据（将字节流转换回Python对象）"""
        if not self.enable_serialization:
            return data.decode()
        return pickle.loads(data)

    def _encode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        编码数据（扩展接口）

        用于在序列化前对数据进行处理，如添加差分隐私噪声、数据压缩或加密等。
        子类可重写此方法实现特定的编码逻辑。
        """
        if not self.enable_serialization:
            return data
        return data

    def _decode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        解码数据（扩展接口）

        用于在反序列化后对数据进行处理，如解密、去噪或数据恢复等。
        子类可重写此方法实现特定的解码逻辑。
        """
        if not self.enable_serialization:
            return data
        return data
