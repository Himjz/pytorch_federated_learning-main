import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import seaborn as sns
from scipy.signal import savgol_filter

# 设置绘图风格 - 学术论文风格
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["pdf.fonttype"] = 42  # 确保PDF中的字体可编辑

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

# 定义学术期刊常用的颜色方案 (适合黑白打印)
JOURNAL_COLORS = [
    '#000000',  # 黑色
    '#E69F00',  # 橙色
    '#56B4E9',  # 蓝色
    '#009E73',  # 绿色
    '#F0E442',  # 黄色
    '#0072B2',  # 深蓝色
    '#D55E00',  # 红色
    '#CC79A7',  # 粉色
    '#999999',  # 灰色
    '#0000FF',  # 纯蓝
]

# 定义线条样式
LINE_STYLES = [
    '-', '--', '-.', ':',
    (0, (3, 1, 1, 1)),  # 密集虚线
    (0, (3, 1, 1, 1, 1, 1)),  # 更密集虚线
    (0, (5, 1)),  # 长虚线
    (0, (5, 1, 1, 1)),  # 虚线点
    (0, (3, 10, 1, 10)),  # 松散虚线
    (0, (3, 10, 1, 10, 1, 10)),  # 松散虚线点
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


def smooth_curve(y, window_size=7, polyorder=3):
    """使用Savitzky-Golay滤波器平滑曲线"""
    if len(y) < window_size:
        return y
    return savgol_filter(y, window_length=window_size, polyorder=polyorder)


def plot_metric(data_dict, metric, figsize=(6.5, 4), output_dir='metric_plots', show=False):
    """
    绘制单个指标的趋势图 - 学术论文版本

    参数:
        data_dict: 包含多个文件数据的字典
        metric: 要绘制的指标名称
        figsize: 图表大小 (英寸)
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

        # 选择颜色和线条样式
        color = JOURNAL_COLORS[i % len(JOURNAL_COLORS)]
        linestyle = LINE_STYLES[i % len(LINE_STYLES)]

        # 绘制平滑曲线
        line, = ax.plot(x, smooth_values, label=label, color=color,
                        linestyle=linestyle, alpha=0.9)

        # 每5个点绘制一个标记，便于区分不同曲线
        if len(values) > 10:
            markers_on = range(1, len(values) + 1, max(1, len(values) // 5))
            ax.plot(np.array(x)[markers_on - 1], np.array(values)[markers_on - 1],
                    marker='o', markersize=4, color=color, alpha=0.9,
                    linestyle='None')

    # 设置图表属性
    ax.set_title(f'{metric_name} Trend', pad=10)
    ax.set_xlabel('Iteration', labelpad=8)
    ax.set_ylabel(f'{metric_name} ({metric_unit})', labelpad=8)

    # 设置坐标轴范围和刻度
    ax.set_xlim(1, max_x)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴只显示整数
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # 设置次要刻度
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # 自定义网格线 - 更轻量，适合学术图表
    ax.grid(True, which='major', linestyle='--', alpha=0.5, color='gray')
    ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='gray')

    # 设置边框
    for spine in ax.spines.values():
        spine.set_color('gray')
        spine.set_linewidth(0.5)

    # 美化图例 - 无框，紧凑布局
    if len(data_dict) > 1:
        legend = ax.legend(frameon=False, loc='best',
                           handlelength=2.5, handletextpad=0.5)

    # 调整布局
    plt.tight_layout(pad=0.5)

    # 保存图表为PDF和PNG格式
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f'{metric}.pdf')
        png_path = os.path.join(output_dir, f'{metric}.png')
        fig.savefig(pdf_path, bbox_inches='tight')
        fig.savefig(png_path, bbox_inches='tight')
        print(f"Saved chart: {pdf_path} and {png_path}")

    # 显示图表
    if show:
        plt.show()

    plt.close(fig)  # 关闭图表以释放内存
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