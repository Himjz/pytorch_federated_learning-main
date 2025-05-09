#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import yaml
from json import JSONEncoder
from tqdm import tqdm

from fed_baselines.client_base import FedClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fednova import FedNovaServer
from fed_baselines.server_shapley import FedShapley

from postprocessing.recorder import Recorder
from preprocessing.self_dataloader import divide_data
from utils.models import *

# 定义 JSON 支持的基本数据类型
json_types = (list, dict, str, int, float, bool, type(None))

class PythonObjectEncoder(JSONEncoder):
    """
    自定义 JSON 编码器，用于处理非 JSON 原生类型的对象。
    将非 JSON 原生类型的对象转换为包含序列化数据的字典，以便进行 JSON 序列化。
    """
    def default(self, obj):
        # 如果对象是 JSON 支持的基本类型，调用父类的 default 方法进行处理
        if isinstance(obj, json_types):
            return super().default(obj)
        # 否则，将对象进行 pickle 序列化，并存储在字典中
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}

def as_python_object(dct):
    """
    自定义 JSON 解码器辅助函数，用于将包含 pickle 序列化数据的字典还原为原始对象。
    :param dct: 待解码的字典
    :return: 解码后的对象或原始字典
    """
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct

def fed_args():
    """
    解析联邦学习基线运行所需的命令行参数。
    :return: 包含命令行参数的对象
    """
    parser = argparse.ArgumentParser()

    # 添加配置文件路径参数，该参数为必需参数
    parser.add_argument('--config', type=str, required=True, help='用于配置的 YAML 文件路径')

    args = parser.parse_args()
    return args

def fed_run():
    """
    联邦学习基线运行的主函数，负责整个联邦学习流程的执行。
    """
    # 解析命令行参数
    args = fed_args()
    # 读取配置文件
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    # 支持的联邦学习算法列表
    algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova", "FedShapley"]
    # 检查配置中的算法是否支持
    assert config["client"]["fed_algo"] in algo_list, "不支持该联邦学习算法"

    # 支持的数据集列表
    dataset_list = ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100', 'SelfDataSet']
    # 检查配置中的数据集是否支持
    assert config["system"]["dataset"] in dataset_list, "不支持该数据集"

    # 支持的模型列表
    model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN"]
    # 检查配置中的模型是否支持
    assert config["system"]["model"] in model_list, "不支持该模型"

    # 设置随机种子，保证结果可复现
    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])

    # 存储客户端实例的字典
    client_dict = {}
    # 初始化记录器，用于记录训练过程中的结果
    recorder = Recorder()

    # 划分训练集和测试集
    trainset_config, testset = divide_data(num_client=config["system"]["num_client"],
                                           num_local_class=config["system"]["num_local_class"],
                                           dataset_name=config["system"]["dataset"],
                                           i_seed=config["system"]["i_seed"])
    # 记录最高准确率
    max_acc = 0
    # 根据联邦学习算法和具体设置初始化客户端
    for client_id in trainset_config['users']:
        if config["client"]["fed_algo"] == 'FedAvg':
            client_dict[client_id] = FedClient(client_id, dataset_id=config["system"]["dataset"],
                                               epoch=config["client"]["num_local_epoch"],
                                               model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            client_dict[client_id] = ScaffoldClient(client_id, dataset_id=config["system"]["dataset"],
                                                    epoch=config["client"]["num_local_epoch"],
                                                    model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedProx':
            client_dict[client_id] = FedProxClient(client_id, dataset_id=config["system"]["dataset"],
                                                   epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedNova':
            client_dict[client_id] = FedNovaClient(client_id, dataset_id=config["system"]["dataset"],
                                                   epoch=config["client"]["num_local_epoch"],
                                                   model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedShapley':
            client_dict[client_id] = FedClient(client_id, dataset_id=config["system"]["dataset"],
                                               epoch=config["client"]["num_local_epoch"],
                                               model_name=config["system"]["model"])
        # 加载客户端的训练集
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # 根据联邦学习算法和具体设置初始化服务器
    if config["client"]["fed_algo"] == 'FedAvg':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                               model_name=config["system"]["model"])
    elif config["client"]["fed_algo"] == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                                    model_name=config["system"]["model"])
        scv_state = fed_server.scv.state_dict()
    elif config["client"]["fed_algo"] == 'FedProx':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                               model_name=config["system"]["model"])
    elif config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                                   model_name=config["system"]["model"])
    elif config["client"]["fed_algo"] == 'FedShapley':
        fed_server = FedShapley(trainset_config['users'], dataset_id=config["system"]["dataset"],
                                model_name=config["system"]["model"])
    # 加载测试集到服务器
    fed_server.load_testset(testset)
    # 获取服务器的全局模型参数
    global_state_dict = fed_server.state_dict()

    # 多轮通信的联邦学习主流程
    pbar = tqdm(range(config["system"]["num_round"]))
    for global_round in pbar:
        for client_id in trainset_config['users']:
            # 本地训练
            if config["client"]["fed_algo"] == 'FedAvg':
                client_dict[client_id].update(global_state_dict)
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
            elif config["client"]["fed_algo"] == 'FedShapley':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)

        # 全局聚合
        fed_server.select_clients()
        if config["client"]["fed_algo"] == 'FedAvg':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()  # SCAFFOLD 算法
        elif config["client"]["fed_algo"] == 'FedProx':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedNova':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedShapley':
            global_state_dict, avg_loss, _ = fed_server.agg()

        # 测试并清空缓存
        accuracy = fed_server.test()
        fed_server.flush()

        # 记录结果
        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)

        # 更新最高准确率
        if max_acc < accuracy:
            max_acc = accuracy
        # 更新进度条显示信息
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
                                                             str(config["system"]["i_seed"])) + '].json', "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)

if __name__ == "__main__":
    # 执行联邦学习主函数
    fed_run()