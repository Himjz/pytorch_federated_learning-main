#!/usr/bin/env python
import argparse
import json
import os
import random
import time
from json import JSONEncoder

import numpy as np
import torch
import yaml
from tqdm import tqdm

from fed_baselines.client_base import FedClient
from fed_baselines.client_feddp import FedDPClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_fednova import FedNovaServer
from fed_baselines.server_scaffold import ScaffoldServer
from postprocessing.recorder import Recorder
from preprocessing.fed_dataloader import UniversalDataLoader

json_types = (list, dict, str, int, float, bool, type(None))

def fed_args():
    """
    联邦学习基线运行所需的参数
    :return: 联邦学习基线的参数
    """
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

    algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova", "FedDp"]
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
    # 新增：记录成员推理攻击准确率
    membership_attack_accuracies = []

    dataloader = UniversalDataLoader(root='./data',
                                                 num_client=config["system"]["num_client"],
                                                 num_local_class=config["system"]["num_local_class"],
                                                 dataset_name=config["system"]["dataset"],
                                                 seed=config["system"]["i_seed"])
    dataloader.load()

    trainset_config, testset = dataloader.divide()

    # 初始化客户端
    for client_id in trainset_config['users']:
        if config["client"]["fed_algo"] == 'FedAvg':
            client_dict[client_id] = FedClient(client_id, epoch=config["client"]["num_local_epoch"],
                                               model_name=config["system"]["model"],
                                               dataset_info=dataloader)
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            client_dict[client_id] = ScaffoldClient(client_id, epoch=config["client"]["num_local_epoch"],
                                                    model_name=config["system"]["model"],
                                                    dataset_info=dataloader)
        elif config["client"]["fed_algo"] == 'FedProx':
            client_dict[client_id] = FedProxClient(client_id, epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"],
                                                   dataset_info=dataloader)
        elif config["client"]["fed_algo"] == 'FedNova':
            client_dict[client_id] = FedNovaClient(client_id, epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"],
                                                   dataset_info=dataloader)
        elif config["client"]["fed_algo"] == 'FedDp':
            client_dict[client_id] = FedDPClient(client_id, epoch=config["client"]["num_local_epoch"],
                                                 model_name=config["system"]["model"],
                                                 dataset_info=dataloader,
                                                 testset=testset)

        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # 初始化服务器
    if config["client"]["fed_algo"] == 'FedAvg':
        fed_server = FedServer(trainset_config['users'], model_name=config["system"]["model"],
                               dataset_info=dataloader)
    elif config["client"]["fed_algo"] == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], model_name=config["system"]["model"],
                                    dataset_info=dataloader)
        scv_state = fed_server.scv.state_dict()
    elif config["client"]["fed_algo"] == 'FedProx':
        fed_server = FedServer(trainset_config['users'], model_name=config["system"]["model"],
                               dataset_info=dataloader)
    elif config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], model_name=config["system"]["model"],
                                   dataset_info=dataloader)
    elif config["client"]["fed_algo"] == 'FedDp':
        fed_server = FedServer(trainset_config['users'], model_name=config["system"]["model"],
                               dataset_info=dataloader)
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
            elif config["client"]["fed_algo"] == 'SCAFFOLD':
                client_dict[client_id].update(global_state_dict, scv_state)
            elif config["client"]["fed_algo"] == 'FedProx':
                client_dict[client_id].update(global_state_dict)
            elif config["client"]["fed_algo"] == 'FedNova':
                client_dict[client_id].update(global_state_dict)
            elif config["client"]["fed_algo"] == 'FedDp':
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
                client_dict[client_id].update(global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, delta_ccv_state)
            elif config["client"]["fed_algo"] == 'FedProx':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == 'FedNova':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, coeff, norm_grad)
            elif config["client"]["fed_algo"] == 'FedDp':
                state_dict, n_data, loss, attack_accuracy = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
                # 新增：记录成员推理攻击准确率
                membership_attack_accuracies.append(attack_accuracy)
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
        elif config["client"]["fed_algo"] == 'FedDp':
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

        pbar.set_description(
            f'Global Round: {global_round} | Train loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}'
        )

        # 保存结果文件（优化命名）
        # 主结果文件
        main_result_path = os.path.join(res_root, f"{base_filename}_result.json")
        with open(main_result_path, "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=JSONEncoder)

        # 时间与指标文件
        time_metrics_path = os.path.join(res_root, f"{base_filename}_time_metrics.json")
        with open(time_metrics_path, "w") as time_jsfile:
            json.dump(time_and_metrics_recorder, time_jsfile, cls=JSONEncoder)

        # 新增：保存成员推理攻击准确率
        membership_attack_path = os.path.join(res_root, f"{base_filename}_membership_attack.json")
        with open(membership_attack_path, "w") as attack_jsfile:
            json.dump(membership_attack_accuracies, attack_jsfile, cls=JSONEncoder)

if __name__ == "__main__":
    fed_run()