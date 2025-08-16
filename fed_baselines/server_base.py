import pickle
import socket
import struct
import threading
import uuid
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score, f1_score, precision_score
from torch.utils.data import DataLoader

from utils.fed_utils import init_model


class FedServer(object):
    def __init__(self, client_list, model_name, dataset_info: list | tuple,
                 server_ip: str = '127.0.0.3', server_port: int = 9999,
                 local: bool = True):
        """
        初始化联邦学习服务器（兼容原始版本接口）
        :param model_name: 模型名称
        :param dataset_info: 数据集信息 (num_class, image_dim, image_channel)
        :param server_ip: 服务器IP地址（网络模式有效）
        :param server_port: 服务器端口（网络模式有效）
        :param local: 是否启用本地模式，默认为True
        """
        # 客户端状态管理 - 保持原始版本数据结构
        self.client_state = {}  # {client_id: state_dict}
        self.client_loss = {}  # {client_id: loss}
        self.client_n_data = {}  # {client_id: n_data}
        self.client_info = {}  # {client_id: info}
        self.selected_clients = []
        self._batch_size = 200
        self.client_list = []

        # 测试数据集
        self.testset = None

        # 联邦学习参数 - 保持原始版本属性
        self.round = 0
        self.n_data = 0

        # 服务器配置
        self.supported_algorithms = ['fedavg', 'scaffold', 'fednova', 'fedprox']
        self.local = local

        # 新增序列化控制参数但保持默认兼容性
        self.local_use_serialization = False  # 默认为False，保持原始行为

        # 轮次管理（网络模式）
        if not self.local:
            self.server_ip = server_ip
            self.server_port = server_port
            self.is_running = False
            self.connection_thread = None
            self.client_handlers = []
            self.round_data = defaultdict(dict)
            self.waiting_clients = defaultdict(list)
            self.round_ready = defaultdict(bool)
        else:
            self.server_ip = None
            self.server_port = None
            self.is_running = False
            self.connection_thread = None
            self.client_handlers = []

        # 设备配置 - 保持原始实现
        gpu = 0
        self._device = torch.device(
            "cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu"
        )

        # 初始化全局模型 - 保持原始逻辑
        self._num_class, self._image_dim, self._image_channel = dataset_info
        self.model_name = model_name
        self.model = init_model(
            model_name=self.model_name,
            num_class=self._num_class,
            image_channel=self._image_channel
        )
        self.global_model_history = {0: self.model.state_dict()}

    # 核心功能 - 保持原始接口
    def load_testset(self, testset):
        self.testset = testset

    def state_dict(self):
        return self.model.state_dict()

    def test(self, default: bool = True):
        if not self.testset:
            raise ValueError("请先加载测试数据集")

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

                test_output = self.model(b_x)
                pred_y = torch.max(test_output, 1)[1].to(self._device).data.squeeze()

                accuracy_collector += (pred_y == b_y).sum().item()
                loss = F.cross_entropy(test_output, b_y)
                loss_collector += loss.item()

                all_preds.extend(pred_y.cpu().numpy())
                all_labels.extend(b_y.cpu().numpy())

        accuracy = accuracy_collector / len(self.testset)
        avg_loss = loss_collector / len(test_loader)

        if default:
            return accuracy
        else:
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            return accuracy, recall, f1, avg_loss, precision

    # 联邦学习流程 - 保持原始行为
    def select_clients(self, connection_ratio=1):
        self.selected_clients = []
        self.n_data = 0

        for client_id in self.client_list:
            b = np.random.binomial(np.ones(1).astype(int), connection_ratio)
            if b:
                self.selected_clients.append(client_id)
                if client_id in self.client_n_data:
                    self.n_data += self.client_n_data[client_id]

    def agg(self):
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        model = init_model(
            model_name=self.model_name,
            num_class=self._num_class,
            image_channel=self._image_channel
        )
        model_state = model.state_dict()
        avg_loss = 0

        for i, client_id in enumerate(self.selected_clients):
            if client_id not in self.client_state:
                continue
            client_weight = self.client_n_data[client_id] / self.n_data

            for key in self.client_state[client_id]:
                if i == 0:
                    model_state[key] = self.client_state[client_id][key] * client_weight
                else:
                    model_state[key] += self.client_state[client_id][key] * client_weight

            avg_loss += self.client_loss[client_id] * client_weight

        self.model.load_state_dict(model_state)
        self.global_model_history[self.round + 1] = model_state
        self.round += 1
        n_data = self.n_data

        return model_state, avg_loss, n_data

    def rec(self, client_id=None, state_dict=None, n_data=None, loss=None, serialized_data=None):
        """接收客户端数据 - 保持原始接口并兼容新功能"""
        # 原始版本行为：本地模式不使用序列化
        if self.local:
            # 兼容原始调用方式：不传入serialized_data
            if serialized_data is not None:
                print("警告: 本地模式下忽略序列化数据（原始版本兼容行为）")
                return False
            # 检查必要参数（原始版本要求）
            if any(param is None for param in [client_id, state_dict, n_data, loss]):
                print("原始参数不完整")
                return False
            return self._process_raw_parameters(client_id, state_dict, n_data, loss)

        # 网络模式处理
        if serialized_data is not None:
            if any(param is not None for param in [client_id, state_dict, n_data, loss]):
                print("不能同时提供序列化数据和原始参数")
                return False
            return self._process_serialized_data(serialized_data)
        else:
            if any(param is None for param in [client_id, state_dict, n_data, loss]):
                print("原始参数不完整")
                return False
            # 网络模式下自动序列化
            serialized = self._serialize({
                'client_id': client_id,
                'model_state': state_dict,
                'n_data': n_data,
                'loss': loss
            })
            encoded = self._encode(serialized)
            return self._process_serialized_data(encoded)

    # 数据处理 - 保持原始逻辑
    def _process_serialized_data(self, encoded_data):
        """处理编码后的序列化数据（网络模式）"""
        try:
            decoded_data = self._decode(encoded_data)
            parameters = self._deserialize(decoded_data)

            required_keys = ['model_state', 'n_data', 'loss', 'client_id']
            if not all(key in parameters for key in required_keys):
                print("上传的参数缺少必要字段")
                return False

            return self._process_raw_parameters(
                client_id=parameters['client_id'],
                state_dict=parameters['model_state'],
                n_data=parameters['n_data'],
                loss=parameters['loss']
            )

        except Exception as e:
            print(f"处理序列化参数时出错: {str(e)}")
            return False

    def _process_raw_parameters(self, client_id, state_dict, n_data, loss):
        """处理原始参数 - 保持原始实现"""
        try:
            if not self.local and client_id not in self.client_list:
                print(f"客户端 {client_id} 未授权上传参数")
                return False

            self.n_data += n_data

            if client_id not in self.client_state:
                self.client_state[client_id] = {}

            self.client_state[client_id].update(state_dict)
            self.client_n_data[client_id] = n_data
            self.client_loss[client_id] = loss

            if client_id in self.client_info:
                self.client_info[client_id]['last_seen'] = datetime.now().isoformat()

            return True

        except Exception as e:
            print(f"处理原始参数时出错: {str(e)}")
            return False

    # 序列化与编码 - 保持向后兼容
    def _serialize(self, data):
        """序列化数据（网络模式使用）"""
        if self.local:
            raise RuntimeError("本地模式下不支持序列化（原始版本兼容行为）")
        try:
            return pickle.dumps(data)
        except Exception as e:
            raise RuntimeError(f"序列化失败: {str(e)}")

    def _deserialize(self, data):
        """反序列化数据（网络模式使用）"""
        if self.local:
            raise RuntimeError("本地模式下不支持反序列化（原始版本兼容行为）")
        try:
            return pickle.loads(data)
        except pickle.UnpicklingError as e:
            raise ValueError(f"反序列化失败: 数据格式不正确 - {str(e)}")
        except Exception as e:
            raise RuntimeError(f"反序列化过程中发生错误: {str(e)}")

    def _encode(self, data):
        """编码数据（为加密扩展设计）"""
        # 原始版本行为：不进行编码处理
        return data

    def _decode(self, data):
        """解码数据（为加密扩展设计）"""
        # 原始版本行为：不进行解码处理
        return data

    # 网络功能 - 保持原始接口
    def start_server(self):
        if self.local:
            raise RuntimeError("本地模式下不支持启动服务器")

        self.is_running = True
        self.connection_thread = threading.Thread(
            target=self._listen_for_connections,
            daemon=True
        )
        self.connection_thread.start()
        print(f"服务器已启动，监听 {self.server_ip}:{self.server_port}")

    def stop_server(self):
        if self.local:
            raise RuntimeError("本地模式下不支持停止服务器")

        self.is_running = False
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join()
        for thread in self.client_handlers:
            if thread.is_alive():
                thread.join()
        print("服务器已停止")

    def _listen_for_connections(self):
        if self.local:
            raise RuntimeError("本地模式下禁用连接监听功能")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((self.server_ip, self.server_port))
                s.listen(5)
                s.settimeout(1.0)

                while self.is_running:
                    try:
                        conn, addr = s.accept()
                        client_thread = threading.Thread(
                            target=self._handle_client_connection,
                            args=(conn, addr),
                            daemon=True
                        )
                        client_thread.start()
                        self.client_handlers.append(client_thread)
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"处理连接时出错: {str(e)}")
                        continue
            except Exception as e:
                print(f"服务器监听出错: {str(e)}")

    def _handle_client_connection(self, conn, addr):
        if self.local:
            raise RuntimeError("本地模式下禁用连接处理功能")

        client_id = None
        try:
            data_len_bytes = conn.recv(4)
            if not data_len_bytes:
                conn.close()
                return

            data_len = struct.unpack('>I', data_len_bytes)[0]
            data = b''
            while len(data) < data_len:
                chunk = conn.recv(min(data_len - len(data), 4096))
                if not chunk:
                    conn.close()
                    return
                data += chunk

            decoded_data = self._decode(data)
            handshake_info = self._deserialize(decoded_data)
            client_id = self._process_handshake(conn, addr, handshake_info)
            if not client_id:
                conn.close()
                return

            while self.is_running and client_id in self.client_list:
                try:
                    msg_len_bytes = conn.recv(4)
                    if not msg_len_bytes:
                        break

                    msg_len = struct.unpack('>I', msg_len_bytes)[0]
                    msg_data = b''
                    while len(msg_data) < msg_len:
                        chunk = conn.recv(min(msg_len - len(msg_data), 4096))
                        if not chunk:
                            break
                        msg_data += chunk

                    if len(msg_data) != msg_len:
                        break

                    decoded_msg = self._decode(msg_data)
                    message = self._deserialize(decoded_msg)

                    if message.get('type') == 'parameters':
                        self.rec(serialized_data=msg_data)
                        conn.sendall(b'RECEIVED')
                    elif message.get('type') == 'wait':
                        self._handle_wait_request(client_id, message['round'], conn)
                    elif message.get('type') == 'get_model':
                        model = self.global_model_history.get(message['round'], self.model.state_dict())
                        response_data = self._serialize({
                            'type': 'model_response',
                            'status': 'success',
                            'model_state': model
                        })
                        encoded_response = self._encode(response_data)
                        conn.sendall(struct.pack('>I', len(encoded_response)) + encoded_response)
                    elif message.get('type') == 'disconnect':
                        break

                except Exception as e:
                    print(f"处理客户端 {client_id} 消息时出错: {str(e)}")
                    break

        finally:
            if client_id and client_id in self.client_list:
                self.client_list.remove(client_id)
            if client_id and client_id in self.client_info:
                del self.client_info[client_id]
            try:
                conn.close()
            except:
                pass

    def _process_handshake(self, conn, addr, handshake_info):
        try:
            client_id = str(uuid.uuid4())
            self.client_info[client_id] = {
                'name': handshake_info['client_name'],
                'ip': handshake_info['client_ip'],
                'port': handshake_info['client_port'],
                'address': addr,
                'timestamp': handshake_info['timestamp'],
                'last_seen': datetime.now().isoformat(),
                'capabilities': handshake_info['capabilities'],
                'socket': conn,
                'round': self.round
            }

            self.client_list.append(client_id)
            print(f"新客户端连接: {handshake_info['client_name']} (ID: {client_id})")

            response_data = self._serialize({
                'status': 'success',
                'client_id': client_id,
                'supported_algorithms': self.supported_algorithms,
                'round': self.round
            })
            encoded_response = self._encode(response_data)
            conn.sendall(struct.pack('>I', len(encoded_response)) + encoded_response)
            return client_id

        except Exception as e:
            print(f"处理握手时出错: {str(e)}")
            error_data = self._serialize({
                'status': 'error',
                'error': str(e)
            })
            encoded_error = self._encode(error_data)
            conn.sendall(struct.pack('>I', len(encoded_error)) + encoded_error)
            return None

    def _handle_wait_request(self, client_id, round_num, conn):
        if self.round_ready.get(round_num, False):
            new_model = self.global_model_history.get(round_num + 1, self.model.state_dict())
            response_data = self._serialize({
                'type': 'new_model',
                'status': 'success',
                'model_state': new_model,
                'round': round_num,
                'next_round': round_num + 1
            })
            encoded_response = self._encode(response_data)
            conn.sendall(struct.pack('>I', len(encoded_response)) + encoded_response)
        else:
            if client_id not in self.waiting_clients[round_num]:
                self.waiting_clients[round_num].append(client_id)
                self.client_info[client_id]['socket'] = conn

    # 通用功能 - 保持原始行为
    def flush(self):
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
        if not self.local:
            self.round_data.clear()
            self.waiting_clients.clear()
            self.round_ready.clear()

    # 新增兼容层：支持原始版本未公开的内部调用模式
    def _handle_local_request(self, client_id, request):
        """处理本地模式下的客户端请求（兼容原始版本交互方式）"""
        try:
            # 模拟网络请求处理
            if isinstance(request, dict) and request.get('type') == 'wait':
                round_num = request['round']
                if self.round_ready.get(round_num, False):
                    new_model = self.global_model_history.get(round_num + 1, self.model.state_dict())
                    return {
                        'type': 'new_model',
                        'status': 'success',
                        'model_state': new_model,
                        'round': round_num,
                        'next_round': round_num + 1
                    }
                else:
                    if client_id not in self.waiting_clients[round_num]:
                        self.waiting_clients[round_num].append(client_id)
                    return {'status': 'waiting', 'round': round_num}
            return {'status': 'error', 'error': '未知请求类型'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
