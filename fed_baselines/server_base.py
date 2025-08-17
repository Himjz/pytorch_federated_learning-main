import pickle
import socket
import threading
import time
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from utils.fed_utils import init_model


class FedServer:
    def __init__(self,
                 client_list: List[str],
                 model_name: str,
                 dataset_info: Union[List, Tuple],
                 device: Optional[str] = None,
                 local: bool = True,
                 ip: str = '0.0.0.0',
                 port: int = 5000,
                 enable_serialization: Optional[bool] = None) -> None:
        # 旧版本兼容属性
        self.client_state: Dict[str, Dict[str, Tensor]] = {}  # 替代 clients_params
        self.client_loss: Dict[str, float] = {}  # 保持名称一致
        self.client_n_data: Dict[str, int] = {}  # 替代 clients_data_size
        self.selected_clients: List[str] = []
        self._batch_size: int = 200  # 测试时的批量大小
        self.client_list: List[str] = client_list
        self.testset: Optional[Dataset] = None  # 旧版本测试集
        self.round: int = 0  # 替代 current_round
        self.n_data: int = 0  # 替代 total_data_size

        # 数据集信息（旧版本参数）
        self._num_class, self._image_dim, self._image_channel = dataset_info

        # 设备选择（兼容旧版本的GPU选择逻辑）
        if device is None:
            gpu = 0  # 旧版本默认GPU编号
            self._device: str = torch.device(
                f"cuda:{gpu}" if torch.cuda.is_available() and gpu != -1 else "cpu"
            ).type
        else:
            self._device = device

        # 模型初始化（使用旧版本的init_model函数）
        self.model_name: str = model_name
        self.model: nn.Module = init_model(
            model_name=self.model_name,
            num_class=self._num_class,
            image_channel=self._image_channel
        ).to(self._device)
        self.global_model: nn.Module = self.model  # 保持与新版本的兼容性

        # 新版本属性
        self.dataset_name: str = ""  # 可根据需要从dataset_info推断
        self.clients_params: Dict[str, Dict[str, Tensor]] = self.client_state  # 兼容新接口
        self.clients_loss: Dict[str, float] = self.client_loss  # 兼容新接口
        self.clients_data_size: Dict[str, int] = self.client_n_data  # 兼容新接口
        self.global_model_history: Dict[int, Dict[str, Tensor]] = {0: self.model.state_dict()}

        # 网络相关属性
        self.local: bool = local
        self.ip: str = ip
        self.port: int = port
        self.server_socket: Optional[socket.socket] = None
        self.running: bool = False
        self.client_info: Dict[str, Tuple[str, int, float]] = {}
        self.supported_algorithms: List[str] = ['fedavg', 'scaffold', 'fedprox']
        self.selected_algorithm: str = 'fedavg'

        # 序列化控制参数
        if enable_serialization is None:
            self.enable_serialization: bool = not self.local
        else:
            self.enable_serialization = enable_serialization if self.local else True

    def load_testset(self, testset: Dataset) -> 'FedServer':
        """兼容旧版本的加载测试集方法"""
        self.testset = testset
        return self

    def state_dict(self) -> Dict[str, Tensor]:
        """旧版本接口：返回全局模型字典"""
        return self.model.state_dict()

    def test(self, default: bool = True) -> Union[float, Tuple[float, float, float, float, float]]:
        """兼容旧版本的测试方法，同时支持新版本的评估指标"""
        if self.testset is None:
            raise ValueError("测试集未加载，请先调用load_testset方法")

        test_loader = DataLoader(self.testset, batch_size=self._batch_size, shuffle=True)
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        total_loss: float = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self._device), y.to(self._device)
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, y)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(test_loader)

        # 兼容旧版本的返回格式
        if default:
            return accuracy
        else:
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            return accuracy, recall, f1, avg_loss, precision

    def select_clients(self, connection_ratio: float = 1.0) -> List[str]:
        """兼容旧版本的客户端选择方法"""
        self.selected_clients = []
        self.n_data = 0

        # 旧版本的二项分布采样方式
        for client_id in self.client_list:
            if np.random.binomial(1, connection_ratio):
                self.selected_clients.append(client_id)
                self.n_data += self.client_n_data.get(client_id, 0)

        return self.selected_clients

    def agg(self) -> Tuple[Dict[str, Tensor], float, int]:
        """兼容旧版本的聚合方法，返回值保持一致"""
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0.0, 0

        # 使用旧版本的模型初始化方式
        model = init_model(
            model_name=self.model_name,
            num_class=self._num_class,
            image_channel=self._image_channel
        )
        model_state = model.state_dict()
        avg_loss: float = 0.0

        # 加权平均聚合（与旧版本逻辑一致）
        for i, client_id in enumerate(self.selected_clients):
            if client_id not in self.client_state:
                continue

            client_params = self.client_state[client_id]
            data_size = self.client_n_data.get(client_id, 0)
            weight = data_size / self.n_data

            for key in model_state:
                if i == 0:
                    model_state[key] = client_params[key] * weight
                else:
                    model_state[key] += client_params[key] * weight

            avg_loss += self.client_loss.get(client_id, 0.0) * weight

        # 更新全局模型
        self.model.load_state_dict(model_state)
        self.round += 1
        self.global_model_history[self.round] = model_state

        return model_state, avg_loss, self.n_data

    def rec(self, name: str, state_dict: Dict[str, Tensor], n_data: int, loss: float) -> None:
        """兼容旧版本的接收客户端数据方法"""
        self.n_data += n_data
        self.client_state[name] = state_dict
        self.client_n_data[name] = n_data
        self.client_loss[name] = loss

        # 网络模式下更新客户端活跃时间
        if not self.local and name in self.client_info:
            self.client_info[name] = (self.client_info[name][0],
                                      self.client_info[name][1],
                                      time.time())

    def flush(self) -> 'FedServer':
        """兼容旧版本的清空客户端状态方法"""
        self.n_data = 0
        self.client_state.clear()
        self.client_n_data.clear()
        self.client_loss.clear()
        self.selected_clients = []
        return self

    # 以下为新版本的网络功能，保持不变
    def start_server(self) -> None:
        if self.local:
            raise ValueError("本地模式下不能启动网络服务器，请设置local=False")

        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(5)
        print(f"服务器启动，监听 {self.ip}:{self.port}")

        threading.Thread(target=self._listen_for_connections, daemon=True).start()

    def stop_server(self) -> None:
        if self.local:
            return

        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("服务器已停止")

    def _listen_for_connections(self) -> None:
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()  # type: ignore
                threading.Thread(target=self._handle_client_connection,
                                 args=(client_socket, addr),
                                 daemon=True).start()
            except Exception as e:
                if self.running:
                    print(f"连接错误: {e}")

    def _handle_client_connection(self, client_socket: socket.socket, addr: Tuple[str, int]) -> None:
        try:
            while self.running:
                data: bytes = client_socket.recv(4096)
                if not data:
                    break

                request: Dict[str, Any]
                if self.enable_serialization:
                    request = self._deserialize(data)
                else:
                    request = data  # type: ignore

                response: Dict[str, Any] = self._process_request(request, addr)

                send_data: Union[bytes, Dict[str, Any]]
                if self.enable_serialization:
                    send_data = self._serialize(response)
                else:
                    send_data = response

                client_socket.sendall(
                    send_data if isinstance(send_data, bytes) else str(send_data).encode())  # type: ignore
        except Exception as e:
            print(f"客户端处理错误 {addr}: {e}")
        finally:
            client_socket.close()

    def _process_request(self, request: Dict[str, Any], addr: Tuple[str, int]) -> Dict[str, Any]:
        if request.get('type') == 'handshake':
            return self._process_handshake(addr)
        elif request.get('type') == 'get_global_model':
            return self._process_model_request(request)
        elif request.get('type') == 'upload_local_model':
            return self._process_model_upload(request)
        else:
            return {'status': 'error', 'message': '未知请求类型'}

    def _process_handshake(self, addr: Tuple[str, int]) -> Dict[str, Any]:
        client_id: str = f"client_{len(self.client_info) + 1}"
        self.client_info[client_id] = (addr[0], addr[1], time.time())
        return {
            'status': 'success',
            'client_id': client_id,
            'supported_algorithms': self.supported_algorithms
        }

    def _process_model_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        client_id: Optional[str] = request.get('client_id')
        round_num: int = request.get('round', self.round)

        if client_id is None or client_id not in self.client_info:
            return {'status': 'error', 'message': '未认证的客户端'}

        if round_num not in self.global_model_history:
            return {'status': 'error', 'message': '无效的轮次号'}

        model_params: Dict[str, Tensor] = self.global_model_history[round_num]
        if self.enable_serialization:
            model_params = self._encode(model_params)

        return {
            'status': 'success',
            'model_params': model_params,
            'current_round': self.round
        }

    def _process_model_upload(self, request: Dict[str, Any]) -> Dict[str, Any]:
        client_id: Optional[str] = request.get('client_id')
        if client_id is None or client_id not in self.client_info:
            return {'status': 'error', 'message': '未认证的客户端'}

        params: Any = request.get('model_params')
        if self.enable_serialization and params is not None:
            params = self._decode(params)

        if isinstance(params, dict) and isinstance(request.get('loss'), (int, float)) and isinstance(
                request.get('data_size'), int):
            self.rec(client_id, params, request['data_size'], request['loss'])
            return {'status': 'success', 'message': '模型已接收'}
        else:
            return {'status': 'error', 'message': '无效的模型参数'}

    # 序列化和编码工具方法
    def _serialize(self, data: Any) -> bytes:
        if not self.enable_serialization:
            return str(data).encode() if not isinstance(data, bytes) else data
        return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> Any:
        if not self.enable_serialization:
            return data.decode()
        return pickle.loads(data)

    def _encode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not self.enable_serialization:
            return data
        return data

    def _decode(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not self.enable_serialization:
            return data
        return data
