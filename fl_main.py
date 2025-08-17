#!/usr/bin/env python
import argparse
import json
import os
import pickle
import random
import time
from json import JSONEncoder

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
from postprocessing.recorder import Recorder
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
    args = parser.parse_args()
    return args


def fed_run():
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova"]
    assert config["client"]["fed_algo"] in algo_list, "The federated learning algorithm is not supported"

    model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN"]
    assert config["system"]["model"] in model_list, "The model is not supported"

    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])

    client_dict = {}
    recorder = Recorder()
    time_and_metrics_recorder = {
        'client_train_avg': [],
        'server_train': [],
        'client_update_avg': [],
        'global_agg': [],
        'model_transfer_avg': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'loss': []
    }

    dataloader = UniversalDataLoader(root='../data',
                                     num_client=config["system"]["num_client"],
                                     num_local_class=config["system"]["num_local_class"],
                                     dataset_name=config["system"]["dataset"],
                                     seed=config["system"]["i_seed"])

    dataloader.load()

    trainset_config, testset = dataloader.divide()
    info = (dataloader.num_classes, dataloader.image_size, dataloader.in_channels)
    max_acc = 0

    # 初始化客户端
    for client_id in trainset_config['users']:
        if config["client"]["fed_algo"] == 'FedAvg':
            client_dict[client_id] = FedClient(client_id, epoch=config["client"]["num_local_epoch"],
                                               model_name=config["system"]["model"],
                                               dataset_info=info)
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            client_dict[client_id] = ScaffoldClient(client_id, epoch=config["client"]["num_local_epoch"],
                                                    model_name=config["system"]["model"],
                                                    dataset_info=info)
        elif config["client"]["fed_algo"] == 'FedProx':
            client_dict[client_id] = FedProxClient(client_id, epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"],
                                                   dataset_info=info)
        elif config["client"]["fed_algo"] == 'FedNova':
            client_dict[client_id] = FedNovaClient(client_id, epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"],
                                                   dataset_info=info)
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # 初始化服务器
    scv_state = None
    if config["client"]["fed_algo"] == 'FedAvg':
        fed_server = FedServer(trainset_config['users'], model_name=config["system"]["model"],
                               dataset_info=info)
    elif config["client"]["fed_algo"] == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], model_name=config["system"]["model"],
                                    dataset_info=info)
        scv_state = fed_server.scv.state_dict()
    elif config["client"]["fed_algo"] == 'FedProx':
        fed_server = FedServer(trainset_config['users'], model_name=config["system"]["model"],
                               dataset_info=info)
    elif config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], model_name=config["system"]["model"],
                                   dataset_info=info)
    else:
        raise ValueError("The federated learning algorithm is not supported")
    fed_server.load_testset(testset)
    global_state_dict = fed_server.state_dict()

    # 生成基础文件名（包含关键参数，简洁易读）
    base_filename = (
        f"{config['client']['fed_algo']}_"
        f"{config['system']['model']}_"
        f"{config['system']['dataset']}_"
        f"clients{config['system']['num_client']}_"
        f"rounds{config['system']['num_round']}_"
        f"epochs{config['client']['num_local_epoch']}_"
        f"seed{config['system']['i_seed']}"
    )
    res_root = config["system"]["res_root"]
    if not os.path.exists(res_root):
        os.makedirs(res_root)

    # 主训练循环
    pbar = tqdm(range(config["system"]["num_round"]))
    for global_round in pbar:
        client_train_times = []
        client_update_times = []
        model_transfer_times = []

        for client_id in trainset_config['users']:
            # 客户端更新模型
            start_update = time.time()
            if config["client"]["fed_algo"] == 'FedAvg':
                client_dict[client_id].update(global_state_dict)
            elif (config["client"]["fed_algo"] == 'SCAFFOLD') & (scv_state is not None):
                client_dict[client_id].update(global_state_dict, scv_state)
            elif config["client"]["fed_algo"] == 'FedProx':
                client_dict[client_id].update(global_state_dict)
            elif config["client"]["fed_algo"] == 'FedNova':
                client_dict[client_id].update(global_state_dict)
            end_update = time.time()
            client_update_times.append(end_update - start_update)

            # 模拟模型传输（下发+上传）
            model_transfer_down_start = time.time()
            model_transfer_down_end = time.time()
            model_transfer_up_start = time.time()
            model_transfer_up_end = time.time()
            model_transfer_times.append(
                (model_transfer_down_end - model_transfer_down_start) +
                (model_transfer_up_end - model_transfer_up_start)
            )

            # 客户端本地训练
            start_train = time.time()
            if config["client"]["fed_algo"] == 'FedAvg':
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == 'SCAFFOLD':
                state_dict, n_data, loss, delta_ccv_state = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, delta_ccv_state)
            elif config["client"]["fed_algo"] == 'FedProx':
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == 'FedNova':
                state_dict, n_data, loss, coeff, norm_grad = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, coeff, norm_grad)
            end_train = time.time()
            client_train_times.append(end_train - start_train)

        # 时间指标计算
        client_train_avg_time = sum(client_train_times) / len(client_train_times)
        client_update_avg_time = sum(client_update_times) / len(client_update_times)
        model_transfer_avg_time = sum(model_transfer_times) / len(model_transfer_times)

        # 服务器操作
        start_server_train = time.time()
        fed_server.select_clients()
        end_server_train = time.time()
        server_train_time = end_server_train - start_server_train

        # 全局聚合
        start_agg = time.time()
        if config["client"]["fed_algo"] == 'FedAvg':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedProx':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedNova':
            global_state_dict, avg_loss, _ = fed_server.agg()
        end_agg = time.time()
        global_agg_time = end_agg - start_agg

        # 测试与指标记录
        accuracy = fed_server.test()
        accuracy_extra, recall, f1, avg_loss, precision = fed_server.test(default=False)
        fed_server.flush()

        time_and_metrics_recorder['client_train_avg'].append(client_train_avg_time)
        time_and_metrics_recorder['server_train'].append(server_train_time)
        time_and_metrics_recorder['client_update_avg'].append(client_update_avg_time)
        time_and_metrics_recorder['global_agg'].append(global_agg_time)
        time_and_metrics_recorder['model_transfer_avg'].append(model_transfer_avg_time)
        time_and_metrics_recorder['accuracy'].append(accuracy_extra)
        time_and_metrics_recorder['precision'].append(precision)
        time_and_metrics_recorder['recall'].append(recall)
        time_and_metrics_recorder['f1'].append(f1)
        time_and_metrics_recorder['loss'].append(avg_loss)

        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)

        if max_acc < accuracy:
            max_acc = accuracy
        pbar.set_description(
            f'Global Round: {global_round} | Train loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Max Acc: {max_acc:.4f}'
        )

        # 保存结果文件（优化命名）
        # 主结果文件
        main_result_path = os.path.join(res_root, f"{base_filename}_result.json")
        with open(main_result_path, "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)

        # 时间与指标文件
        time_metrics_path = os.path.join(res_root, f"{base_filename}_time_metrics.json")
        with open(time_metrics_path, "w") as time_jsfile:
            json.dump(time_and_metrics_recorder, time_jsfile, cls=PythonObjectEncoder)


if __name__ == "__main__":
    fed_run()
