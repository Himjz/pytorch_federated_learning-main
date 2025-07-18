import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import seaborn as sns
from scipy.signal import savgol_filter

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["axes.grid.axis"] = "y"  # 只显示y轴网格线
plt.rcParams["grid.color"] = "#eeeeee"  # 浅灰色网格线
plt.rcParams["grid.linewidth"] = 0.8  # 较细的网格线

# 定义常用指标的英文名称映射
METRIC_NAMES = {
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1-Score',
    'loss': 'Loss',
    'client_train_avg': 'Client Avg. Training Time',
    'server_train': 'Server Training Time',
    'client_update_avg': 'Client Avg. Update Time',
    'global_agg': 'Global Aggregation Time',
    'model_transfer_avg': 'Model Transfer Avg. Time'
}

# 定义常用指标的单位映射
METRIC_UNITS = {
    'accuracy': '%',
    'precision': '%',
    'recall': '%',
    'f1': '%',
    'loss': 'Loss',
    'client_train_avg': 's',
    'server_train': 's',
    'client_update_avg': 's',
    'global_agg': 's',
    'model_transfer_avg': 's'
}

# 专业配色方案
PROFESSIONAL_COLORS = [
    '#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974',
    '#64B5CD', '#4878CF', '#6ACC65', '#D65F5F', '#B47CC7',
    '#C4AD66', '#77BEDB', '#3B75AF', '#38A649', '#A63838',
    '#7660B0', '#B2984A', '#3A98B8'
]


def load_json_files(file_paths):
    """
    加载多个JSON文件的数据

    参数:
        file_paths: JSON文件路径列表

    返回:
        字典，键为文件名（不含扩展名），值为文件中的数据
    """
    data_dict = {}
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 提取文件名作为键
            file_name = os.path.splitext(os.path.basename(path))[0]
            data_dict[file_name] = data
            print(f"Successfully loaded file: {file_name}")
        except Exception as e:
            print(f"Failed to load file {path}: {str(e)}")
    return data_dict


def get_common_metrics(data_dict):
    """获取所有文件共有的指标"""
    if not data_dict:
        return []
    # 取第一个文件的指标作为基准
    first_data = next(iter(data_dict.values()))
    common_metrics = set(first_data.keys())

    # 求所有文件指标的交集
    for data in data_dict.values():
        common_metrics.intersection_update(data.keys())

    # 按预设顺序排序，其余指标放在后面
    priority_order = ['accuracy', 'precision', 'recall', 'f1', 'loss',
                      'client_train_avg', 'server_train', 'client_update_avg',
                      'global_agg', 'model_transfer_avg']
    sorted_metrics = []
    # 先添加优先级高的指标
    for metric in priority_order:
        if metric in common_metrics:
            sorted_metrics.append(metric)
            common_metrics.remove(metric)
    # 再添加剩余指标
    sorted_metrics.extend(sorted(common_metrics))
    return sorted_metrics


def get_metric_display_name(metric):
    """获取指标的显示名称"""
    return METRIC_NAMES.get(metric, metric.replace('_', ' ').title())


def get_metric_unit(metric):
    """获取指标的单位"""
    return METRIC_UNITS.get(metric, '')


def smooth_curve(y, window_size=5, polyorder=3):
    """使用Savitzky-Golay滤波器平滑曲线"""
    if len(y) < window_size:
        return y
    return savgol_filter(y, window_length=window_size, polyorder=polyorder)


def plot_metric(data_dict, metric, figsize=(12, 8), output_dir='metric_plots', show=False):
    """
    绘制单个指标的趋势图

    参数:
        data_dict: 包含多个文件数据的字典
        metric: 要绘制的指标名称
        figsize: 图表大小
        output_dir: 图表保存目录
        show: 是否显示图表

    返回:
        图表对象
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 获取指标的显示名称和单位
    metric_name = get_metric_display_name(metric)
    metric_unit = get_metric_unit(metric)

    max_x = 0
    for i, (label, data) in enumerate(data_dict.items()):
        if metric not in data:
            continue

        values = np.array(data[metric])
        x = np.arange(1, len(values) + 1)
        max_x = max(max_x, len(values))

        # 计算平滑曲线
        try:
            smooth_values = smooth_curve(values)
        except:
            smooth_values = values  # 如果平滑失败，使用原始值

        # 设置线条样式
        line_style = '-'
        alpha = 0.9
        line_width = 2.5

        # 绘制平滑曲线
        line, = ax.plot(x, smooth_values, label=label, color=PROFESSIONAL_COLORS[i % len(PROFESSIONAL_COLORS)],
                        linewidth=line_width, alpha=alpha)

        # 绘制原始数据点，设置适当的透明度和大小
        ax.scatter(x, values, s=25, color=PROFESSIONAL_COLORS[i % len(PROFESSIONAL_COLORS)], alpha=0.4,
                   edgecolors='none', zorder=3)

    # 设置图表属性
    ax.set_title(f'{metric_name} Trend Analysis', fontsize=16, pad=15, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=14, labelpad=10)
    ax.set_ylabel(f'{metric_name} ({metric_unit})', fontsize=14, labelpad=10)

    # 设置坐标轴范围和刻度
    # 扩展X轴范围，防止数据点落在边缘
    x_padding = max(1, max_x * 0.05)  # 至少扩展1个单位，或5%的总范围
    ax.set_xlim(1 - x_padding, max_x + x_padding)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴只显示整数
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # 设置次要刻度

    # 设置y轴范围，留出适当空间
    y_min, y_max = ax.get_ylim()
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)

    # 自定义网格线
    ax.grid(True, which='major', linestyle='-', alpha=0.5, color='#eeeeee')

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 设置坐标轴边框
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.5)

    # 美化图例
    legend = ax.legend(fontsize=12, frameon=True, loc='best',
                       fancybox=True, framealpha=0.9, shadow=False,
                       edgecolor='#dddddd')
    legend.get_frame().set_linewidth(0.5)

    # 添加水平参考线（如果有意义）
    if metric in ['accuracy', 'precision', 'recall', 'f1']:
        ax.axhline(y=50, color='#cccccc', linestyle='--', linewidth=1, alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{metric}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        print(f"Saved chart: {save_path}")

    # 显示图表
    if show:
        plt.show()

    return fig


def main():
    # 自动获取当前目录下的所有JSON文件
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in the current directory.")
        return

    print(f"Found {len(json_files)} JSON files.")

    # 加载所有JSON文件
    data_dict = load_json_files(json_files)
    if not data_dict:
        print("Failed to load any data files.")
        return

    # 获取所有公共指标
    metrics = get_common_metrics(data_dict)
    print(f"Metrics to plot: {', '.join([get_metric_display_name(m) for m in metrics])}")

    # 为每个指标绘制图表
    for metric in metrics:
        try:
            plot_metric(data_dict, metric, show=False)
        except Exception as e:
            print(f"Error plotting {get_metric_display_name(metric)}: {str(e)}")

    print("All charts generated successfully.")


if __name__ == "__main__":
    main()