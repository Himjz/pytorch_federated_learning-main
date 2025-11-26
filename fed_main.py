#!/usr/bin/env python
import argparse
import json
import os
import pickle
import random
import socket
import threading
import time
from json import JSONEncoder
from typing import Literal

import numpy as np
import torch
import yaml
from tqdm import tqdm

from fed_baselines.client_base import FedClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_fednova import FedNovaServer
from fed_baselines.server_scaffold import ScaffoldServer

from preprocessing.fed_dataloader import UniversalDataLoader

json_types = (list, dict, str, int, float, bool, type(None))


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


def fed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='用于配置的 YAML 文件')
    parser.add_argument('--identity', type=str, choices=['client', 'server'], required=True,
                        help='部署身份：client（客户端）或 server（服务器）')
    parser.add_argument('--local-test', action='store_true', default=True,
                        help='是否启用本地模拟模式（True）或实际网络传输（False）')
    parser.add_argument('--server-ip', type=str, default='127.0.0.1',
                        help='服务器IP地址')
    parser.add_argument('--server-port', type=int, default=8080,
                        help='服务器端口号')
    parser.add_argument('--con-time', type=int, default=None,
                        help='最长等待时间(秒)，超时后无论客户端数量多少都开始训练，不指定则使用配置文件值')
    args = parser.parse_args()
    return args


# 网络传输工具函数
def send_data(sock, data):
    """发送数据通过网络，带长度前缀"""
    try:
        serialized = pickle.dumps(data)
        sock.sendall(len(serialized).to_bytes(4, byteorder='big'))
        sock.sendall(serialized)
        return True
    except Exception as e:
        print(f"发送数据错误: {e}")
        return False


def receive_data(sock):
    """接收网络数据，带长度前缀解析"""
    try:
        length_bytes = sock.recv(4)
        if not length_bytes:
            return None
        length = int.from_bytes(length_bytes, byteorder='big')
        data = b''
        while len(data) < length:
            chunk = sock.recv(min(length - len(data), 4096))
            if not chunk:
                return None
            data += chunk
        return pickle.loads(data)
    except Exception as e:
        print(f"接收数据错误: {e}")
        return None


class FedMain:
    """
    联邦学习主流程封装类，保持与原始代码接口一致：
    - 从配置文件读取预设客户端数量（兼容原始配置）
    - 达到预设数量或超时后自动开始训练
    - 动态适应实际连接的客户端数量
    """

    def __init__(self, config,
                 identity: Literal['client', 'server'],
                 local_test: bool = True,
                 server_ip: str = '127.0.0.1',
                 server_port: int = 8080,
                 con_time: int = None):
        # 部署配置参数（与原始接口保持一致）
        self.identity = identity
        self.local_test = local_test
        self.server_ip = server_ip
        self.server_port = server_port

        # 从配置文件读取预设客户端数量（兼容原始配置结构）
        # 优先使用num_client作为预设客户端数量，保持与原始代码兼容
        self.preset_clients = config["system"].get("num_client", 5)
        # 最长等待时间：命令行参数优先，其次使用配置文件，最后使用默认值
        self.con_time = con_time or config["system"].get("con_time", 60)

        # 基础配置与校验（保持原始校验逻辑）
        self.config = config
        self._validate_config()

        # 随机种子初始化（保持原始实现）
        self._init_seed()

        # 核心组件初始化（与原始接口保持一致）
        self.client_dict = {}
        self.fed_server = None
        self.dataloader = None
        self.time_metrics = self._init_time_metrics()

        # 网络与连接管理（增强功能但保持接口兼容）
        self.socket = None
        self.client_sockets = {}  # {client_id: socket}
        self.client_ip_map = {}  # {client_id: ip_address} IP与ID映射
        self.connected = False
        self.connection_thread = None
        self.accept_connections = True  # 控制是否接受新连接

        # 客户端ID管理（保持原始命名习惯）
        self.next_client_id = 1  # 用于分配客户端ID
        self.num_clients = 0  # 实际客户端数量（动态获取）

        # 关键状态变量（与原始接口保持一致）
        self.global_state_dict = None
        self.scv_state = None
        self.max_acc = 0
        self.base_filename = self._generate_base_filename()
        self.res_root = config["system"]["res_root"]
        self._init_result_dir()

    def _validate_config(self):
        """校验配置文件合法性（保持原始校验逻辑）"""
        algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova"]
        model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN"]

        assert self.config["client"]["fed_algo"] in algo_list, f"不支持的联邦算法，可选：{algo_list}"
        assert self.config["system"]["model"] in model_list, f"不支持的模型，可选：{model_list}"
        assert self.preset_clients > 0, "预设客户端数量必须大于0"
        assert self.con_time > 0, "最长等待时间必须大于0"

    def _init_seed(self):
        """初始化所有随机种子（保持原始实现）"""
        seed = self.config["system"]["i_seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def _init_time_metrics(self):
        """初始化时间与性能指标记录字典（保持原始结构）"""
        return {
            'client_train_avg': [],
            'server_train': [],
            'client_update_avg': [],
            'global_agg': [],
            'model_transfer_avg': [],
            'global_model_transfer': [],
            'network_latency': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'loss': []
        }

    def _generate_base_filename(self):
        """生成包含关键参数的基础文件名（保持原始命名格式）"""
        client_suffix = f"_clients{self.num_clients}" if self.num_clients > 0 else ""
        return (
            f"{self.identity}_"
            f"{self.config['client']['fed_algo']}_"
            f"{self.config['system']['model']}_"
            f"{self.config['system']['dataset']}_"
            f"rounds{self.config['system']['num_round']}_"
            f"mode_{'local' if self.local_test else 'network'}"
            f"{client_suffix}"
        )

    def _init_result_dir(self):
        """初始化结果保存目录（保持原始路径结构）"""
        mode_dir = os.path.join(self.res_root, 'local_mode' if self.local_test else 'network_mode')
        identity_dir = os.path.join(mode_dir, self.identity)
        if not os.path.exists(identity_dir):
            os.makedirs(identity_dir)
        self.res_root = identity_dir

    # ------------------------------
    # 连接管理（增强功能，不改变外部接口）
    # ------------------------------
    def connection_listener(self):
        """
        连接监听线程函数
        接受客户端连接并分配ID，满足以下任一条件时停止：
        1. 连接的客户端数量达到预设值（从配置文件num_client读取）
        2. 等待时间超过最长等待时间
        """
        start_time = time.time()
        print(f"开始接受客户端连接（预设数量: {self.preset_clients}个，最长等待: {self.con_time}秒）...")

        while self.accept_connections:
            # 检查是否达到预设客户端数量
            if len(self.client_sockets) >= self.preset_clients:
                print(f"已达到预设客户端数量 ({self.preset_clients}个)，停止等待")
                break

            # 检查是否超时
            elapsed_time = time.time() - start_time
            if elapsed_time > self.con_time:
                print(f"超过最长等待时间 ({self.con_time}秒)，停止等待")
                break

            try:
                # 设置超时，定期检查退出条件
                self.socket.settimeout(1.0)
                client_sock, client_addr = self.socket.accept()
                client_ip = client_addr[0]

                # 分配客户端ID
                client_id = f"client_{self.next_client_id}"
                self.next_client_id += 1

                # 记录连接信息
                self.client_sockets[client_id] = client_sock
                self.client_ip_map[client_id] = client_ip
                self.num_clients = len(self.client_sockets)

                # 向客户端发送分配的ID
                send_data(client_sock, {
                    'type': 'id_assignment',
                    'client_id': client_id,
                    'message': f"已分配客户端ID: {client_id}，当前连接数: {self.num_clients}/{self.preset_clients}"
                })

                print(f"客户端 {client_id} 从 {client_ip} 连接，当前连接数: {self.num_clients}/{self.preset_clients}")
                print(f"剩余等待时间: {max(0, int(self.con_time - elapsed_time))}秒")

            except socket.timeout:
                continue  # 超时继续等待
            except Exception as e:
                print(f"接受连接错误: {e}")
                continue

        # 连接阶段结束
        self.accept_connections = False
        self.num_clients = len(self.client_sockets)
        print(f"客户端连接阶段结束，共连接 {self.num_clients} 个客户端")

    def start(self):
        """
        服务器启动函数（保持原始接口）
        启动连接监听线程，等待客户端连接，满足条件后开始联邦学习流程
        """
        if self.identity != 'server':
            return

        # 启动连接监听线程
        self.connection_thread = threading.Thread(target=self.connection_listener, daemon=True)
        self.connection_thread.start()

        # 等待连接线程完成
        self.connection_thread.join()

        # 确保至少有一个客户端连接
        if self.num_clients == 0:
            print("没有客户端连接，无法启动联邦学习")
            self.close_network()
            return

        # 更新配置中的客户端数量（覆盖配置文件）
        self.config["system"]["num_client"] = self.num_clients
        print(f"更新配置：客户端数量 = {self.num_clients}")

        # 初始化数据和服务器组件
        self._init_federation()

        # 开始主循环
        self.main_loop()

    def _init_federation(self):
        """初始化联邦学习所需的数据和组件（保持原始接口）"""
        # 加载数据
        if self.local_test:
            trainset_config, testset = self.setup_dataloader()
        else:
            trainset_config, testset = self.setup_dataloader()

        dataset_info = (self.dataloader.num_classes, self.dataloader.image_size, self.dataloader.in_channels)

        # 初始化服务器
        self.setup_server(trainset_config['users'], dataset_info, testset)

        # 本地模式下同时初始化客户端
        if self.local_test:
            self.setup_clients(trainset_config, dataset_info)

    def setup_network(self):
        """初始化网络连接（保持原始接口）"""
        if self.local_test:
            print(f"本地模拟模式，创建 {self.preset_clients} 个虚拟客户端连接")

            # 本地模式下，创建预设数量的虚拟客户端
            self.num_clients = self.preset_clients
            for i in range(1, self.num_clients + 1):
                client_id = f"client_{i}"
                self.client_sockets[client_id] = f"virtual_socket_{client_id}"
                self.client_ip_map[client_id] = f"127.0.0.{i}"

            self.connected = True
            return

        try:
            if self.identity == 'server':
                # 服务器创建监听套接字
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.bind((self.server_ip, self.server_port))
                self.socket.listen(self.preset_clients)  # 最多监听预设数量的客户端
                print(f"服务器启动，监听 {self.server_ip}:{self.server_port}")
                self.connected = True

            else:
                # 客户端连接到服务器并获取ID
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.server_ip, self.server_port))
                print(f"客户端已连接到服务器 {self.server_ip}:{self.server_port}，等待分配ID...")

                # 接收服务器分配的ID
                data = receive_data(self.socket)
                if data and data['type'] == 'id_assignment':
                    self.client_id = data['client_id']
                    print(f"客户端ID分配成功: {self.client_id}")
                    self.connected = True
                else:
                    print("未能获取客户端ID，连接失败")
                    self.connected = False

        except Exception as e:
            print(f"网络连接错误: {e}")
            self.connected = False

    def close_network(self):
        """关闭网络连接（保持原始接口）"""
        if self.local_test or not self.connected:
            return

        try:
            if self.identity == 'server':
                for sock in self.client_sockets.values():
                    sock.close()
                if self.socket:
                    self.socket.close()
                print("服务器关闭所有连接")
            else:
                if self.socket:
                    self.socket.close()
                print(f"客户端 {self.client_id} 关闭连接")
        except Exception as e:
            print(f"关闭网络连接错误: {e}")
        finally:
            self.connected = False

    # ------------------------------
    # 数据与组件初始化（保持原始接口）
    # ------------------------------
    def setup_dataloader(self):
        """初始化并加载联邦数据集（保持原始接口）"""
        # 本地模式和网络模式均使用实际连接的客户端数量
        num_client = self.num_clients if self.num_clients > 0 else self.preset_clients

        self.dataloader = UniversalDataLoader(
            root='../data',
            num_client=num_client,
            num_local_class=self.config["system"]["num_local_class"],
            dataset_name=self.config["system"]["dataset"],
            seed=self.config["system"]["i_seed"]
        )
        self.dataloader.load()

        trainset_config, testset = self.dataloader.divide()

        if not self.local_test and self.identity == 'client':
            # 网络模式客户端只保留自己的训练数据
            client_data = {
                'users': [self.client_id],
                'user_data': {
                    self.client_id: trainset_config['user_data'][self.client_id]
                }
            }
            return client_data, None

        return trainset_config, testset

    def setup_clients(self, trainset_config, dataset_info):
        """初始化客户端实例（保持原始接口）"""
        if self.identity != 'client' and not self.local_test:
            return

        algo = self.config["client"]["fed_algo"]
        local_epoch = self.config["client"]["num_local_epoch"]
        model_name = self.config["system"]["model"]

        client_ids = trainset_config['users']
        for client_id in client_ids:
            # 根据算法选择客户端类型
            if algo == 'FedAvg':
                client = FedClient(client_id, epoch=local_epoch, model_name=model_name, dataset_info=dataset_info)
            elif algo == 'SCAFFOLD':
                client = ScaffoldClient(client_id, epoch=local_epoch, model_name=model_name, dataset_info=dataset_info)
            elif algo == 'FedProx':
                client = FedProxClient(client_id, epoch=local_epoch, model_name=model_name, dataset_info=dataset_info)
            elif algo == 'FedNova':
                client = FedNovaClient(client_id, epoch=local_epoch, model_name=model_name, dataset_info=dataset_info)
            else:
                raise ValueError(f"不支持的联邦算法：{algo}")

            # 加载客户端本地训练集
            client.load_trainset(trainset_config['user_data'][client_id])
            self.client_dict[client_id] = client

    def setup_server(self, user_list, dataset_info, testset):
        """初始化服务器实例（保持原始接口）"""
        if self.identity != 'server':
            return

        algo = self.config["client"]["fed_algo"]
        model_name = self.config["system"]["model"]

        # 根据算法选择服务器类型
        if algo == 'FedAvg' or algo == 'FedProx':
            self.fed_server = FedServer(user_list, model_name=model_name, dataset_info=dataset_info)
        elif algo == 'SCAFFOLD':
            self.fed_server = ScaffoldServer(user_list, model_name=model_name, dataset_info=dataset_info)
            self.scv_state = self.fed_server.scv.state_dict()
        elif algo == 'FedNova':
            self.fed_server = FedNovaServer(user_list, model_name=model_name, dataset_info=dataset_info)
        else:
            raise ValueError(f"不支持的联邦算法：{algo}")

        # 加载服务器测试集并初始化全局模型状态
        self.fed_server.load_testset(testset)
        self.global_state_dict = self.fed_server.state_dict()

    # ------------------------------
    # 传输过程实现（保持原始接口）
    # ------------------------------
    def transfer_global_model_to_client(self, client_id, global_state_dict):
        """服务器向客户端传输全局模型（保持原始接口）"""
        start_time = time.time()
        processed_state = global_state_dict

        if not self.local_test:
            if self.identity == 'server':
                # 服务器发送模型到指定客户端
                send_data(self.client_sockets[client_id], {
                    'type': 'global_model',
                    'data': global_state_dict,
                    'scv_state': self.scv_state if self.config["client"]["fed_algo"] == 'SCAFFOLD' else None
                })
            else:
                # 客户端接收模型
                data = receive_data(self.socket)
                if data and data['type'] == 'global_model':
                    processed_state = data['data']
                    if self.config["client"]["fed_algo"] == 'SCAFFOLD':
                        self.scv_state = data['scv_state']

        transfer_time = time.time() - start_time
        return processed_state, transfer_time

    def transfer_client_model_to_server(self, client_id, client_state_dict, extra_data=None):
        """客户端向服务器传输本地模型（保持原始接口）"""
        start_time = time.time()
        extra_data = extra_data or {}
        processed_state = client_state_dict

        if not self.local_test:
            if self.identity == 'client':
                # 客户端发送本地模型
                send_data(self.socket, {
                    'type': 'client_model',
                    'client_id': client_id,
                    'data': client_state_dict,
                    'extra': extra_data
                })
            else:
                # 服务器在单独的接收逻辑中处理
                pass

        transfer_time = time.time() - start_time
        return processed_state, transfer_time

    def broadcast_global_model(self, global_state_dict):
        """服务器广播全局模型到所有客户端（保持原始接口）"""
        start_time = time.time()

        if not self.local_test and self.identity == 'server':
            # 网络模式：向所有客户端广播
            for client_id, sock in self.client_sockets.items():
                send_data(sock, {
                    'type': 'broadcast_model',
                    'data': global_state_dict
                })

        broadcast_time = time.time() - start_time
        return broadcast_time

    # ------------------------------
    # 核心流程函数（保持原始接口）
    # ------------------------------
    def run_client_stage(self, client_id):
        """客户端阶段处理（保持原始接口）"""
        if self.identity != 'client' and not self.local_test:
            return 0, 0, 0, {}

        client = self.client_dict.get(client_id)
        if not client:
            return 0, 0, 0, {}

        algo = self.config["client"]["fed_algo"]
        extra_data = {}

        # 1. 接收全局模型
        processed_global_state, down_transfer_time = self.transfer_global_model_to_client(
            client_id, self.global_state_dict)

        # 2. 客户端更新模型
        start_update = time.time()
        if algo == 'SCAFFOLD' and self.scv_state is not None:
            client.update(processed_global_state, self.scv_state)
        else:
            client.update(processed_global_state)
        client_update_time = time.time() - start_update

        # 3. 客户端本地训练
        start_train = time.time()
        if algo == 'FedAvg' or algo == 'FedProx':
            state_dict, n_data, loss = client.train()
        elif algo == 'SCAFFOLD':
            state_dict, n_data, loss, delta_ccv_state = client.train()
            extra_data['delta_ccv_state'] = delta_ccv_state
        elif algo == 'FedNova':
            state_dict, n_data, loss, coeff, norm_grad = client.train()
            extra_data['coeff'] = coeff
            extra_data['norm_grad'] = norm_grad
        client_train_time = time.time() - start_train

        # 4. 上传本地模型
        processed_local_state, up_transfer_time = self.transfer_client_model_to_server(
            client_id, state_dict, extra_data)
        model_transfer_time = down_transfer_time + up_transfer_time

        # 本地模式下直接记录到服务器
        if self.local_test and self.identity == 'server':
            if algo == 'FedAvg' or algo == 'FedProx':
                self.fed_server.rec(client.name, processed_local_state, n_data, loss)
            elif algo == 'SCAFFOLD':
                self.fed_server.rec(client.name, processed_local_state, n_data, loss, extra_data['delta_ccv_state'])
            elif algo == 'FedNova':
                self.fed_server.rec(client.name, processed_local_state, n_data, loss,
                                    extra_data['coeff'], extra_data['norm_grad'])

        return client_train_time, client_update_time, model_transfer_time, {
            'n_data': n_data,
            'loss': loss,
            'extra': extra_data
        }

    def run_server_stage(self):
        """服务器阶段处理（保持原始接口）"""
        if self.identity != 'server':
            return None, 0, 0, 0, 0, {}

        algo = self.config["client"]["fed_algo"]

        # 1. 服务器选择客户端
        start_server_train = time.time()
        selected_clients = self.fed_server.select_clients()
        server_train_time = time.time() - start_server_train

        # 2. 接收客户端模型（仅网络模式需要）
        if not self.local_test:
            for _ in range(len(selected_clients)):
                # 从任意客户端接收数据
                for client_id, sock in self.client_sockets.items():
                    data = receive_data(sock)
                    if data and data['type'] == 'client_model':
                        client_id = data['client_id']
                        state_dict = data['data']
                        extra = data['extra']
                        n_data = extra['n_data']
                        loss = extra['loss']

                        if algo == 'FedAvg' or algo == 'FedProx':
                            self.fed_server.rec(client_id, state_dict, n_data, loss)
                        elif algo == 'SCAFFOLD':
                            self.fed_server.rec(client_id, state_dict, n_data, loss, extra['delta_ccv_state'])
                        elif algo == 'FedNova':
                            self.fed_server.rec(client_id, state_dict, n_data, loss,
                                                extra['coeff'], extra['norm_grad'])
                        break

        # 3. 全局模型聚合
        start_agg = time.time()
        if algo == 'SCAFFOLD':
            global_state_dict, avg_loss, _, self.scv_state = self.fed_server.agg()
        else:
            global_state_dict, avg_loss, _ = self.fed_server.agg()
        global_agg_time = time.time() - start_agg

        # 4. 广播全局模型
        broadcast_time = self.broadcast_global_model(global_state_dict)

        # 5. 模型性能测试
        accuracy = self.fed_server.test()
        accuracy_extra, recall, f1, avg_loss, precision = self.fed_server.test(default=False)
        self.fed_server.flush()

        # 更新最大准确率
        if self.max_acc < accuracy:
            self.max_acc = accuracy

        metrics = {
            'accuracy': accuracy,
            'accuracy_extra': accuracy_extra,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_loss': avg_loss
        }

        return global_state_dict, avg_loss, server_train_time, global_agg_time, broadcast_time, metrics

    def record_metrics(self, client_train_avg, client_update_avg, model_transfer_avg,
                       server_train, global_agg, broadcast_time, metrics):
        """记录本轮时间指标与性能指标（保持原始接口）"""
        # 时间指标
        if self.identity == 'client' or self.local_test:
            self.time_metrics['client_train_avg'].append(client_train_avg)
            self.time_metrics['client_update_avg'].append(client_update_avg)
            self.time_metrics['model_transfer_avg'].append(model_transfer_avg)

        if self.identity == 'server' or self.local_test:
            self.time_metrics['server_train'].append(server_train)
            self.time_metrics['global_agg'].append(global_agg)
            self.time_metrics['global_model_transfer'].append(broadcast_time)

        # 性能指标
        if self.identity == 'server' or self.local_test:
            self.time_metrics['accuracy'].append(metrics['accuracy_extra'])
            self.time_metrics['precision'].append(metrics['precision'])
            self.time_metrics['recall'].append(metrics['recall'])
            self.time_metrics['f1'].append(metrics['f1'])
            self.time_metrics['loss'].append(metrics['avg_loss'])



    def save_results(self):
        """保存当前所有结果到JSON文件（保持原始接口）"""
        # 动态更新文件名（包含实际客户端数量）
        current_filename = self._generate_base_filename()



        # 时间与指标文件
        time_metrics_path = os.path.join(self.res_root, f"{current_filename}_time_metrics.json")
        with open(time_metrics_path, "w") as f:
            json.dump(self.time_metrics, f, cls=PythonObjectEncoder)

    # ------------------------------
    # 主循环（保持原始接口）
    # ------------------------------
    def main_loop(self):
        """联邦学习主循环（保持原始接口）"""
        try:
            if self.identity == 'client' and not self.local_test:
                # 网络模式客户端初始化
                trainset_config, _ = self.setup_dataloader()
                dataset_info = (self.dataloader.num_classes, self.dataloader.image_size, self.dataloader.in_channels)
                self.setup_clients(trainset_config, dataset_info)

            # 多轮训练循环
            num_round = self.config["system"]["num_round"]
            pbar = tqdm(range(num_round))
            for global_round in pbar:
                client_train_avg = 0
                client_update_avg = 0
                model_transfer_avg = 0

                if self.identity == 'client' or self.local_test:
                    # 客户端处理
                    client_train_times = []
                    client_update_times = []
                    model_transfer_times = []

                    client_ids = list(self.client_dict.keys()) if self.client_dict else [self.client_id]
                    for client_id in client_ids:
                        train_t, update_t, transfer_t, _ = self.run_client_stage(client_id)
                        client_train_times.append(train_t)
                        client_update_times.append(update_t)
                        model_transfer_times.append(transfer_t)

                    # 计算客户端平均时间
                    if client_train_times:
                        client_train_avg = sum(client_train_times) / len(client_train_times)
                        client_update_avg = sum(client_update_times) / len(client_update_times)
                        model_transfer_avg = sum(model_transfer_times) / len(model_transfer_times)

                # 服务器处理
                if self.identity == 'server' or self.local_test:
                    (self.global_state_dict, avg_loss, server_train_t,
                     agg_t, broadcast_t, metrics) = self.run_server_stage()

                    # 记录指标
                    self.record_metrics(client_train_avg, client_update_avg, model_transfer_avg,
                                        server_train_t, agg_t, broadcast_t, metrics)

                    # 更新进度条
                    pbar.set_description(
                        f'Round: {global_round} | Loss: {avg_loss:.4f} | Acc: {metrics["accuracy"]:.4f} | Max Acc: {self.max_acc:.4f}'
                    )
                else:
                    # 网络模式客户端记录
                    self.record_metrics(client_train_avg, client_update_avg, model_transfer_avg,
                                        0, 0, 0, {})
                    pbar.set_description(f'Client {self.client_id} Round: {global_round}')

                # 保存本轮结果
                self.save_results()

        finally:
            # 清理资源
            self.close_network()


def fed_run():
    # 解析配置文件和部署参数（保持原始接口）
    args = fed_args()
    with open(args.config, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"配置文件解析错误: {exc}")
            return

    # 初始化联邦学习主对象（保持原始接口参数）
    fed_main = FedMain(
        config=config,
        identity=args.identity,
        local_test=args.local_test,
        server_ip=args.server_ip,
        server_port=args.server_port,
        con_time=args.con_time
    )

    # 启动网络连接（保持原始调用方式）
    if fed_main.identity == 'server':
        # 服务器模式：先设置网络，再启动（start函数包含连接管理）
        fed_main.setup_network()
        if fed_main.connected:
            fed_main.start()
    else:
        # 客户端模式：设置网络后直接进入主循环
        fed_main.setup_network()
        if fed_main.connected:
            fed_main.main_loop()


if __name__ == "__main__":
    fed_run()
