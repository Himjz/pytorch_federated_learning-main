#!/usr/bin/env python
import argparse
import json
import os
import pickle
import random
import time
from json import JSONEncoder

import yaml
from tqdm import tqdm

from fed_baselines.client_base import FedClient
from fed_baselines.client_feddp1 import FedDPClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_fednova import FedNovaServer
from fed_baselines.server_scaffold import ScaffoldServer
from postprocessing.recorder import Recorder
from preprocessing.fed_dataloader import divide_data
from utils.models import *

json_types = (list, dict, str, int, float, bool, type(None))


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(obj)  # 修正调用错误
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


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
    """
    联邦学习基线的主函数
    """
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova", 'FedDp']
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

    trainset_config, testset, info = divide_data(root='./data',
                                                 num_client=config["system"]["num_client"],
                                                 num_local_class=config["system"]["num_local_class"],
                                                 dataset_name=config["system"]["dataset"],
                                                 i_seed=config["system"]["i_seed"],
                                                 print_report=True,
                                                 untrusted_strategies=None)
    max_acc = 0
    # 根据联邦学习算法和特定的联邦设置初始化客户端
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
        elif config["client"]["fed_algo"] == 'FedDp':
            client_dict[client_id] = FedDPClient(client_id, dataset_id=config["system"]["dataset"],
                                               epoch=config["client"]["num_local_epoch"],
                                               model_name=config["system"]["model"],
                                                 dataset_info=info)
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # 根据联邦学习算法和特定的联邦设置初始化服务器
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
    elif config["client"]["fed_algo"] == 'FedDp':
        fed_server = FedServer(trainset_config['users'], model_name=config["system"]["model"],
                                               dataset_info=info)
    fed_server.load_testset(testset)
    global_state_dict = fed_server.state_dict()

    # 多轮通信的联邦学习主流程

    pbar = tqdm(range(config["system"]["num_round"]))
    for global_round in pbar:
        client_train_times = []
        client_update_times = []
        model_transfer_times = []

        for client_id in trainset_config['users']:
            # 记录客户端更新模型时间
            start_update = time.time()
            if config["client"]["fed_algo"] == 'FedAvg':
                client_dict[client_id].update(global_state_dict)
            elif config["client"]["fed_algo"] == 'SCAFFOLD':
                client_dict[client_id].update(global_state_dict, scv_state)
            elif config["client"]["fed_algo"] == 'FedProx':
                client_dict[client_id].update(global_state_dict)
            elif config["client"]["fed_algo"] == 'FedNova':
                client_dict[client_id].update(global_state_dict)
            end_update = time.time()
            client_update_times.append(end_update - start_update)

            # 记录模型传输下发时间
            model_transfer_down_start = time.time()
            # 模拟下发模型时间
            model_transfer_down_end = time.time()

            # 记录客户端训练时间
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
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            end_train = time.time()
            client_train_times.append(end_train - start_train)

            # 记录模型传输上传时间
            model_transfer_up_start = time.time()
            # 模拟上传模型时间
            model_transfer_up_end = time.time()
            model_transfer_times.append(
                model_transfer_down_end - model_transfer_down_start +
                model_transfer_up_end - model_transfer_up_start
            )

        # 全局聚合
        # 计算各项平均时间
        client_train_avg_time = sum(client_train_times) / len(client_train_times)
        client_update_avg_time = sum(client_update_times) / len(client_update_times)
        model_transfer_avg_time = sum(model_transfer_times) / len(model_transfer_times)

        # 记录服务器训练时间
        start_server_train = time.time()
        fed_server.select_clients()
        end_server_train = time.time()
        server_train_time = end_server_train - start_server_train

        # 记录全局聚合时间
        start_agg = time.time()
        if config["client"]["fed_algo"] == 'FedAvg':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()  # SCAFFOLD 算法
        elif config["client"]["fed_algo"] == 'FedProx':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedNova':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config['client']['fed_algo'] == 'FedDp' :
            global_state_dict, avg_loss, _ = fed_server.agg()
        end_agg = time.time()
        global_agg_time = end_agg - start_agg

        # 测试与刷新
        accuracy = fed_server.test()
        accuracy_extra, recall, f1, avg_loss, precision = fed_server.test(default=False)
        fed_server.flush()

        # 记录时间和指标
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

        # 记录原有结果
        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)

        if max_acc < accuracy:
            max_acc = accuracy
        pbar.set_description(
            'Global Round: %d' % global_round +
            '| Train loss: %.4f ' % avg_loss +
            '| Accuracy: %.4f' % accuracy +
            '| Max Acc: %.4f' % max_acc)

        # 保存结果
        if not os.path.exists(config["system"]["res_root"]):
            os.makedirs(config["system"]["res_root"])

        with open(os.path.join(config["system"]["res_root"], '[\'%s\',' % config["client"]["fed_algo"] +
                                                             '\'%s\',' % config["system"]["model"] +
                                                             str(config["client"]["num_local_epoch"]) + ',' +
                                                             str(config["system"]["num_local_class"]) + ',' +
                                                             str(config["system"]["i_seed"])) + '].json',
                  "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)

        # 保存时间和指标到新的 JSON 文件
        time_metrics_filename = os.path.join(
            config["system"]["res_root"],
            f'[\'{config["client"]["fed_algo"]}\','
            f'\'{config["system"]["model"]}\','
            f'{config["client"]["num_local_epoch"]},'
            f'{config["system"]["num_local_class"]},'
            f'{config["system"]["i_seed"]}]_time_metrics.json'
        )
        with open(time_metrics_filename, "w") as time_jsfile:
            json.dump(time_and_metrics_recorder, time_jsfile, cls=PythonObjectEncoder)


if __name__ == "__main__":
    fed_run()
