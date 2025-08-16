import pickle
import socket
import threading
import time
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class FedServer:
    def __init__(self,
                 client_list:List[str],
                 model_name: str,
                 dataset_name: str,
                 device: str = 'cpu',
                 local: bool = True,
                 ip: str = '0.0.0.0',
                 port: int = 5000,
                 enable_serialization: Optional[bool] = None) -> None:
        self.client_list = client_list
        self.model_name: str = model_name
        self.dataset_name: str = dataset_name
        self.device: str = device
        self.clients_params: Dict[str, Dict[str, Tensor]] = {}  # 客户端模型参数
        self.clients_loss: Dict[str, float] = {}  # 客户端损失
        self.clients_data_size: Dict[str, int] = {}  # 客户端数据量
        self.selected_clients: List[str] = []  # 选中的客户端
        self.total_data_size: int = 0  # 总数据量
        self.test_loader: Optional[DataLoader[Dataset]] = None  # 测试数据集加载器

        # 网络相关属性
        self.local: bool = local  # 是否为本地模式
        self.ip: str = ip
        self.port: int = port
        self.server_socket: Optional[socket.socket] = None
        self.running: bool = False
        self.client_info: Dict[str, Tuple[str, int, float]] = {}  # 客户端信息 {client_id: (ip, port, last_active)}
        self.global_model: nn.Module = self._init_global_model()  # 全局模型
        self.global_model_history: Dict[int, Dict[str, Tensor]] = {0: self.global_model.state_dict()}  # 模型历史
        self.current_round: int = 0
        self.supported_algorithms: List[str] = ['fedavg', 'scaffold', 'fedprox']  # 支持的联邦算法
        self.selected_algorithm: str = 'fedavg'  # 默认算法

        # 序列化控制参数
        # 网络模式下强制开启序列化，本地模式默认关闭
        if enable_serialization is None:
            self.enable_serialization: bool = not self.local
        else:
            # 网络模式下忽略用户设置，强制开启
            self.enable_serialization = enable_serialization if self.local else True

    def _init_global_model(self) -> nn.Module:
        """初始化全局模型"""
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

    def load_testset(self, test_loader: DataLoader[Dataset]) -> 'FedServer':
        """加载测试集（与0.py接口一致）"""
        self.test_loader = test_loader
        return self

    def test(self) -> Dict[str, float]:
        """测试当前全局模型（与0.py接口一致）"""
        if self.test_loader is None:
            raise ValueError("测试集未加载，请先调用load_testset方法")

        self.global_model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        total_loss: float = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.global_model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
            'f1': f1_score(all_labels, all_preds, average='macro'),
            'loss': total_loss / len(self.test_loader)
        }

    def select_clients(self, client_ids: List[str], fraction: float = 0.5) -> List[str]:
        """选择客户端（与0.py接口一致）"""
        if not client_ids:
            return []

        num_select: int = max(1, int(len(client_ids) * fraction))
        self.selected_clients = np.random.choice(client_ids, num_select, replace=False).tolist()
        self.total_data_size = sum(self.clients_data_size.get(cid, 0) for cid in self.selected_clients)
        return self.selected_clients

    def agg(self) -> Tuple[Dict[str, Tensor], float]:
        """聚合客户端模型（与0.py接口一致）"""
        if not self.selected_clients or self.total_data_size == 0:
            return self.global_model.state_dict(), 0.0

        global_params: Dict[str, Tensor] = self.global_model.state_dict()
        for key in global_params:
            global_params[key] = torch.zeros_like(global_params[key], device=self.device)

        # 加权平均聚合
        for cid in self.selected_clients:
            client_params = self.clients_params.get(cid)
            data_size = self.clients_data_size.get(cid, 0)

            if client_params is None or data_size == 0:
                continue

            weight: float = data_size / self.total_data_size
            for key in global_params:
                global_params[key] += client_params[key].to(self.device) * weight

        # 更新全局模型
        self.global_model.load_state_dict(global_params)
        self.current_round += 1
        self.global_model_history[self.current_round] = global_params

        # 计算平均损失
        avg_loss: float = sum(self.clients_loss.get(cid, 0.0) for cid in self.selected_clients) / len(
            self.selected_clients)
        return global_params, avg_loss

    def rec(self, client_id: str, params: Union[Dict[str, Tensor], bytes], loss: float, data_size: int) -> None:
        """接收客户端数据（与0.py接口一致）"""
        # 根据序列化设置处理参数
        if self.enable_serialization:
            processed_params: Dict[str, Tensor]
            if isinstance(params, bytes):
                deserialized = self._deserialize(params)
                processed_params = self._decode(deserialized)
            else:
                processed_params = self._decode(params)  # type: ignore
        else:
            processed_params = params if isinstance(params, dict) else {}  # type: ignore

        self.clients_params[client_id] = processed_params
        self.clients_loss[client_id] = loss
        self.clients_data_size[client_id] = data_size

        # 网络模式下更新客户端活跃时间
        if not self.local and client_id in self.client_info:
            self.client_info[client_id] = (self.client_info[client_id][0],
                                           self.client_info[client_id][1],
                                           time.time())

    def flush(self) -> 'FedServer':
        """清空客户端状态（与0.py接口一致）"""
        self.clients_params.clear()
        self.clients_loss.clear()
        self.selected_clients = []
        self.total_data_size = 0
        return self

    # 网络功能
    def start_server(self) -> None:
        """启动服务器（网络模式）"""
        if self.local:
            raise ValueError("本地模式下不能启动网络服务器，请设置local=False")

        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(5)
        print(f"服务器启动，监听 {self.ip}:{self.port}")

        # 启动监听线程
        threading.Thread(target=self._listen_for_connections, daemon=True).start()

    def stop_server(self) -> None:
        """停止服务器（网络模式）"""
        if self.local:
            return

        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("服务器已停止")

    def _listen_for_connections(self) -> None:
        """监听客户端连接"""
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
        """处理客户端连接"""
        try:
            while self.running:
                data: bytes = client_socket.recv(4096)
                if not data:
                    break

                # 反序列化请求
                request: Dict[str, Any]
                if self.enable_serialization:
                    request = self._deserialize(data)
                else:
                    request = data  # type: ignore

                response: Dict[str, Any] = self._process_request(request, addr)

                # 发送响应
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
        """处理客户端请求"""
        if request.get('type') == 'handshake':
            return self._process_handshake(addr)
        elif request.get('type') == 'get_global_model':
            return self._process_model_request(request)
        elif request.get('type') == 'upload_local_model':
            return self._process_model_upload(request)
        else:
            return {'status': 'error', 'message': '未知请求类型'}

    def _process_handshake(self, addr: Tuple[str, int]) -> Dict[str, Any]:
        """处理客户端握手"""
        client_id: str = f"client_{len(self.client_info) + 1}"
        self.client_info[client_id] = (addr[0], addr[1], time.time())
        return {
            'status': 'success',
            'client_id': client_id,
            'supported_algorithms': self.supported_algorithms
        }

    def _process_model_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理模型请求"""
        client_id: Optional[str] = request.get('client_id')
        round_num: int = request.get('round', self.current_round)

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
            'current_round': self.current_round
        }

    def _process_model_upload(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理模型上传"""
        client_id: Optional[str] = request.get('client_id')
        if client_id is None or client_id not in self.client_info:
            return {'status': 'error', 'message': '未认证的客户端'}

        # 处理上传的模型参数
        params: Any = request.get('model_params')
        if self.enable_serialization and params is not None:
            params = self._decode(params)

        # 调用rec方法接收参数（保持接口兼容）
        if isinstance(params, dict) and isinstance(request.get('loss'), (int, float)) and isinstance(
                request.get('data_size'), int):
            self.rec(client_id, params, request['loss'], request['data_size'])
            return {'status': 'success', 'message': '模型已接收'}
        else:
            return {'status': 'error', 'message': '无效的模型参数'}

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
