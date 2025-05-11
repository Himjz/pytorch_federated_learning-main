import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from pathlib import Path

# 设置中文字体
mpl.rcParams["font.family"] = ["SimHei",]
mpl.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义指标名称的中文映射
chinese_mapping = {
    "client_train_avg": "客户端平均训练时间",
    "server_train": "服务器训练时间",
    "client_update_avg": "客户端平均更新时间",
    "global_agg": "全局聚合时间",
    "model_transfer_avg": "模型平均传输时间",
    "accuracy": "准确率",
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1 分数",
    "local_train_time": "本地训练时间",
    "model_transfer_time": "模型传输时间",
    "global_agg_time": "全局聚合时间",
    "model_update_time": "模型更新时间",
    "train_loss": "训练损失",
    "loss": "损失"
}

# 定义时间指标列表
time_metrics = [
    "client_train_avg", "server_train", "client_update_avg",
    "global_agg", "model_transfer_avg", "local_train_time",
    "model_transfer_time", "global_agg_time", "model_update_time"
]

# 定义损失指标列表
loss_metrics = [
    "train_loss", "test_loss"
]

# 定义多种颜色（每种颜色对应一个文件）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

# 定义要读取的JSON文件目录和文件模式
json_dir = ".."  # 当前目录
file_pattern = "*.json"  # 匹配所有JSON文件

# 获取所有匹配的JSON文件
json_files = ["Shapley-LeNet.json"]


# 从文件名中提取算法名称（用于图例）
def extract_algorithm(filename):
    try:
        # 从"算法类型.json"格式中提取算法名称
        algorithm = Path(filename).stem
        return algorithm
    except:
        return f"文件{Path(filename).name}"


# 读取所有JSON文件的数据
all_data = {}
for i, file_path in enumerate(json_files):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                algorithm = extract_algorithm(file_path)
                all_data[algorithm] = data
                print(f"已加载: {algorithm} ({file_path})")
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
    else:
        print(f"文件不存在: {file_path}")

# 如果没有数据，退出程序
if not all_data:
    print("没有找到有效的数据文件")
    exit()

# 获取所有可能的指标名称
all_metrics = set()
for algorithm, data in all_data.items():
    all_metrics.update(data.keys())

# 为每个指标绘制合并图表
for metric in all_metrics:
    # 创建图形
    plt.figure(figsize=(14, 9))

    # 记录是否有数据可绘制
    has_data = False

    # 遍历所有算法的数据
    for i, (algorithm, data) in enumerate(all_data.items()):
        if metric in data:
            values = data[metric]
            color = colors[i % len(colors)]

            # 绘制折线图，添加算法名称到图例
            plt.plot(values, color=color, linewidth=2.5, marker='o', markersize=4,
                     markevery=max(1, len(values) // 10), label=algorithm)

            has_data = True

    if not has_data:
        plt.close()
        continue

    # 设置标题和坐标轴标签
    title = chinese_mapping.get(metric, metric)
    plt.title(f'{title} 对比分析', fontsize=22, fontweight='bold')
    plt.xlabel('训练轮次', fontsize=16)

    # 根据指标类型设置y轴标签
    if metric in time_metrics:
        plt.ylabel('时间 (秒)', fontsize=16)
    elif metric in loss_metrics:
        plt.ylabel('损失值', fontsize=16)
    else:
        plt.ylabel('指标值', fontsize=16)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置坐标轴刻度字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 添加图例（显示不同文件的来源）
    plt.legend(fontsize=14, loc='best')

    # 添加统计信息（只显示第一个文件的，避免重复）
    if metric in time_metrics and metric in all_data[list(all_data.keys())[0]]:
        values = all_data[list(all_data.keys())[0]][metric]
        stats = f'最大值: {max(values):.3f}\n最小值: {min(values):.3f}\n平均值: {np.mean(values):.3f}'
    elif metric in loss_metrics and metric in all_data[list(all_data.keys())[0]]:
        values = all_data[list(all_data.keys())[0]][metric]
        stats = f'最大值: {max(values):.3f}\n最小值: {min(values):.3f}'
    else:
        stats = ""

    if stats:
        plt.text(0.95, 0.05, stats,
                 transform=plt.gca().transAxes,
                 fontsize=12,
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 优化布局
    plt.tight_layout()

    # 保存图表（可选）
    # plt.savefig(f'{metric}_comparison.png', dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()