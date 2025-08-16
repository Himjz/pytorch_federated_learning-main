import pickle
import socket
import struct
import threading
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.fed_utils import init_model


class FedClient(object):
    def __init__(self, client_name: str, model_name: str, dataset_info: tuple,
                 server_ip: str = '127.0.0.1', server_port: int = 9999,
                 local: bool = True):  # 保持v1版本默认本地模式
        """
        初始化联邦        :param client_name: 客户端名称
        :param model_name: 模型名称
        :param dataset_info: 数据集信息 (num_class, image_dim, image_channel)
        :param server_ip: 服务器IP
        :param server_port: 服务器端口
        :param local: 是否本地模式
        """
        # 保持v1版本的初始化参数和顺序
        self.client_name = client_name
        self.model_name = model_name
        self.local = local  # 默认本地模式

        # 服务器信息
        self.server_ip = server_ip
        self.server_port = server_port
        self.server_info = None

        # 客户端ID
        self.client_id = None

        # 数据和模型
        self.trainset = None
        self.valset = None
        self._num_class, self._image_dim, self._image_channel = dataset_info
        self.model = init_model(
            model_name=self.model_name,
            num_class=self._num_class,
            image_channel=self._image_channel
        )
        self._batch_size = 64
        self._local_epochs = 1

        # 训练状态 - 保持v1版本的属性名
        self.current_round = 0
        self.total_rounds = 0

        # 设备配置
        gpu = 0
        self._device = torch.device(
            "cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu"
        )

        # 网络连接
        self.socket = None
        self.is_connected = False
        self.event = threading.Event()  # 用于等待服务器响应

        # 客户端能力
        self.capabilities = {
            'algorithms': ['fedavg', 'scaffold', 'fednova', 'fedprox'],
            'compression': False,
            'encryption': False,
            'max_batch_size': 256
        }

    def load_data(self, trainset, valset=None):
        """加载数据集 - 保持v1接口"""
        self.trainset = trainset
        self.valset = valset

    def set_train_parameters(self, batch_size: int = 64, local_epochs: int = 1, total_rounds: int = 10):
        """设置训练参数 - 保持v1接口"""
        self._batch_size = batch_size
        self._local_epochs = local_epochs
        self.total_rounds = total_rounds

    def train(self, model_params: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float, int]:
        """本地训练 - 保持v1接口和返回值"""
        # 加载全局模型参数
        self.model.load_state_dict(model_params)
        self.model.to(self._device)
        self.model.train()

        # 准备数据加载器
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # 本地训练
        total_loss = 0.0
        for epoch in range(self._local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self._device), target.to(self._device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        # 返回模型参数、损失和数据量 - 保持v1返回格式
        return self.model.state_dict(), avg_loss, len(self.trainset)

    # 数据序列化/反序列化 - 保持v1接口
    def _serialize(self, data: Any) -> bytes:
        return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

    def _encode(self, data: bytes) -> bytes:
        return data

    # 网络相关方法 - 保持v1接口
    def connect(self) -> bool:
        """与服务器建立连接并完成握手"""
        if self.local:
            raise RuntimeError("本地模式下禁用网络连接功能")

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            print(f"已连接到服务器 {self.server_ip}:{self.server_port}")

            if self._perform_handshake():
                self.is_connected = True
                return True
            else:
                self.socket.close()
                self.socket = None
                return False

        except Exception as e:
            print(f"连接服务器失败: {str(e)}")
            self.socket = None
            return False

    def _perform_handshake(self) -> bool:
        """执行握手 - 保持v1接口"""
        if self.local:
            raise RuntimeError("本地模式下禁用握手功能")

        try:
            # 构建握手请求 - 兼容v1格式
            handshake_request = {
                'client_name': self.client_name,
                'timestamp': datetime.now().isoformat(),
                'capabilities': self.capabilities,
                'client_ip': self.socket.getsockname()[0],
                'client_port': self.socket.getsockname()[1]
            }

            # 序列化并发送
            request_data = self._serialize(handshake_request)
            encoded_data = self._encode(request_data)
            self.socket.sendall(struct.pack('>I', len(encoded_data)) + encoded_data)

            # 接收响应
            response_len_bytes = self.socket.recv(4)
            if not response_len_bytes:
                print("未收到服务器握手响应长度")
                return False

            response_len = struct.unpack('>I', response_len_bytes)[0]
            response_data = b''
            while len(response_data) < response_len:
                chunk = self.socket.recv(min(response_len - len(response_data), 4096))
                if not chunk:
                    return False
                response_data += chunk

            # 处理响应
            decoded_response = self._encode(response_data)
            handshake_response = self._deserialize(decoded_response)

            if 'status' in handshake_response and handshake_response['status'] == 'success':
                self.client_id = handshake_response['client_id']
                self.current_round = handshake_response.get('round', 0)
                self.server_info = {
                    'supported_algorithms': handshake_response.get('supported_algorithms', []),
                    'total_rounds': handshake_response.get('total_rounds', self.total_rounds)
                }
                print(f"握手成功，客户端ID: {self.client_id}")
                return True
            else:
                error_msg = handshake_response.get('error', '未知错误')
                print(f"握手失败: {error_msg}")
                return False

        except Exception as e:
            print(f"握手过程出错: {str(e)}")
            return False

    def run_federated_training(self) -> bool:
        """运行联邦学习流程 - 保持v1接口"""
        if self.local:
            # 本地模式逻辑保持不变
            print("本地模式下运行联邦训练")
            for round_num in range(self.total_rounds):
                self.current_round = round_num
                print(f"\n===== 开始本地训练轮次 {self.current_round + 1}/{self.total_rounds} =====")
                model_params, loss, n_data = self.train(self.model.state_dict())
                self.model.load_state_dict(model_params)
                print(f"轮次 {self.current_round + 1} 训练完成，损失: {loss:.4f}")
            return True

        if not self.is_connected:
            print("未连接到服务器，尝试连接...")
            if not self.connect():
                return False

        try:
            # 按轮次进行训练 - 保持v1流程
            while self.current_round < self.total_rounds:
                print(f"\n===== 开始联邦训练轮次 {self.current_round + 1}/{self.total_rounds} =====")

                # 1. 获取当前全局模型
                print("获取全局模型...")
                global_model = self.get_global_model()
                if not global_model:
                    print("获取全局模型失败，终止训练")
                    return False

                # 2. 本地训练
                print("进行本地训练...")
                local_model, loss, n_data = self.train(global_model)

                # 3. 上传本地模型参数
                print("上传本地模型参数...")
                if not self.send_parameters(local_model, loss, n_data):
                    print("上传参数失败，终止训练")
                    return False

                # 4. 发送等待请求
                print("等待服务器聚合模型...")
                if not self.send_wait_request():
                    print("发送等待请求失败，终止训练")
                    return False

                # 5. 等待服务器响应
                self.event.clear()
                print("等待服务器响应...")

                # 启动线程监听响应
                response_thread = threading.Thread(target=self._wait_for_server_response)
                response_thread.start()
                self.event.wait()

                # 检查新模型
                if not hasattr(self, '_new_global_model') or not self._new_global_model:
                    print("未收到新的全局模型，终止训练")
                    return False

                # 更新轮次
                self.current_round += 1
                print(f"轮次 {self.current_round} 完成")

            print("\n===== 所有联邦训练轮次完成 =====")
            return True

        except Exception as e:
            print(f"联邦训练过程出错: {str(e)}")
            self.close()
            return False

    def send_parameters(self, model_state: Dict[str, torch.Tensor], loss: float, n_data: int) -> bool:
        """发送参数 - 保持v1接口"""
        if self.local:
            raise RuntimeError("本地模式下禁用网络发送功能")

        if not self.is_connected or not self.socket:
            print("未连接到服务器，请先调用connect()")
            return False

        try:
            # 保持v1的数据格式
            params = {
                'type': 'parameters',
                'client_id': self.client_id,
                'round': self.current_round,
                'model_state': model_state,
                'loss': loss,
                'n_data': n_data,
                'timestamp': datetime.now().isoformat()
            }

            # 发送数据
            data = self._serialize(params)
            encoded_data = self._encode(data)
            self.socket.sendall(struct.pack('>I', len(encoded_data)) + encoded_data)

            # 接收确认
            response = self.socket.recv(1024)
            if response == b'RECEIVED':
                print("参数已成功发送并被服务器接收")
                return True
            else:
                print(f"服务器拒绝接收参数，响应: {response.decode()}")
                return False

        except Exception as e:
            print(f"发送参数时出错: {str(e)}")
            self.is_connected = False
            self.socket.close()
            self.socket = None
            return False

    def send_wait_request(self) -> bool:
        """发送等待请求 - 保持v1接口"""
        if self.local:
            raise RuntimeError("本地模式下禁用网络发送功能")

        if not self.is_connected or not self.socket:
            print("未连接到服务器，请先调用connect()")
            return False

        try:
            # 保持v1的数据格式
            request = {
                'type': 'wait',
                'client_id': self.client_id,
                'round': self.current_round,
                'timestamp': datetime.now().isoformat()
            }

            data = self._serialize(request)
            encoded_data = self._encode(data)
            self.socket.sendall(struct.pack('>I', len(encoded_data)) + encoded_data)
            return True

        except Exception as e:
            print(f"发送等待请求时出错: {str(e)}")
            return False

    def _wait_for_server_response(self):
        """等待服务器响应 - 内部方法保持兼容"""
        try:
            # 接收响应长度
            response_len_bytes = self.socket.recv(4)
            if not response_len_bytes:
                print("未收到服务器响应长度")
                self.event.set()
                return

            response_len = struct.unpack('>I', response_len_bytes)[0]
            response_data = b''
            while len(response_data) < response_len:
                chunk = self.socket.recv(min(response_len - len(response_data), 4096))
                if not chunk:
                    print("未收到完整的服务器响应")
                    self.event.set()
                    return
                response_data += chunk

            # 处理响应
            decoded_response = self._encode(response_data)
            response = self._deserialize(decoded_response)

            if 'type' in response and response['type'] == 'new_model' and response['status'] == 'success':
                print("收到新的全局模型")
                self._new_global_model = response['model_state']
                self.model.load_state_dict(self._new_global_model)
            else:
                print(f"服务器响应错误: {response.get('error', '未知错误')}")
                self._new_global_model = None

        except Exception as e:
            print(f"等待服务器响应时出错: {str(e)}")
            self._new_global_model = None
        finally:
            self.event.set()

    def get_global_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取全局模型 - 保持v1接口"""
        if self.local:
            return self.model.state_dict()

        if not self.is_connected or not self.socket:
            print("未连接到服务器，请先调用connect()")
            return None

        try:
            # 保持v1的数据格式
            request = {
                'type': 'get_model',
                'client_id': self.client_id,
                'round': self.current_round,
                'timestamp': datetime.now().isoformat()
            }

            data = self._serialize(request)
            encoded_data = self._encode(data)
            self.socket.sendall(struct.pack('>I', len(encoded_data)) + encoded_data)

            # 接收模型
            model_len_bytes = self.socket.recv(4)
            if not model_len_bytes:
                print("未收到模型长度")
                return None

            model_len = struct.unpack('>I', model_len_bytes)[0]
            model_data = b''
            while len(model_data) < model_len:
                chunk = self.socket.recv(min(model_len - len(model_data), 4096))
                if not chunk:
                    print("未收到完整模型数据")
                    return None
                model_data += chunk

            decoded_model = self._encode(model_data)
            model_state = self._deserialize(decoded_model)
            return model_state

        except Exception as e:
            print(f"获取全局模型时出错: {str(e)}")
            self.is_connected = False
            self.socket.close()
            self.socket = None
            return None

    def close(self):
        """关闭连接 - 保持v1接口"""
        if self.local:
            raise RuntimeError("本地模式下禁用网络关闭功能")

        if self.socket:
            try:
                disconnect_msg = self._serialize({
                    'type': 'disconnect',
                    'client_id': self.client_id,
                    'timestamp': datetime.now().isoformat()
                })
                self.socket.sendall(struct.pack('>I', len(disconnect_msg)) + disconnect_msg)
                self.socket.close()
            except Exception as e:
                print(f"关闭连接时出错: {str(e)}")
            self.is_connected = False
            self.socket = None
            print("已断开与服务器的连接")
