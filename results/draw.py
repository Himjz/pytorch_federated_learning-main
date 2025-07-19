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
plt.rcParams["legend.loc"] = "upper left"  # 默认图例位置，避免遮挡数据

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

# 扩展专业配色方案（支持15组数据）
PROFESSIONAL_COLORS = [
    '#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974',
    '#64B5CD', '#4878CF', '#6ACC65', '#D65F5F', '#B47CC7',
    '#C4AD66', '#77BEDB', '#3B75AF', '#38A649', '#A63838'
]

# 线条样式组合（用于区分15组数据）
LINE_STYLES = [
    ('-', 'o'), ('-', 's'), ('-', '^'), ('-', 'D'), ('-', '*'),
    ('--', 'o'), ('--', 's'), ('--', '^'), ('--', 'D'), ('--', '*'),
    ('-.', 'o'), ('-.', 's'), ('-.', '^'), (':', 'o'), (':', 's')
]


def load_json_files(file_paths):
    """加载多个JSON文件的数据"""
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


def plot_metric(data_dict, metric, figsize=(14, 9), output_dir='metric_plots', show=False):
    """绘制单个指标的趋势图，优化支持15组数据"""
    # 创建图表（增大尺寸以适应更多数据）
    fig, ax = plt.subplots(figsize=figsize)

    # 获取指标的显示名称和单位
    metric_name = get_metric_display_name(metric)
    metric_unit = get_metric_unit(metric)

    max_x = 0
    num_datasets = len(data_dict)

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

        # 获取线条样式和标记（支持15组数据区分）
        line_style, marker = LINE_STYLES[i % len(LINE_STYLES)]
        color = PROFESSIONAL_COLORS[i % len(PROFESSIONAL_COLORS)]

        # 绘制平滑曲线
        line, = ax.plot(x, smooth_values, label=label, color=color,
                        linestyle=line_style, linewidth=2.2, alpha=0.9)

        # 绘制原始数据点（根据数据量调整大小）
        marker_size = 40 if len(x) < 20 else 30
        ax.scatter(x, values, s=marker_size, color=color, alpha=0.6,
                   marker=marker, edgecolors='white', linewidth=0.5, zorder=3)

    # 设置图表属性
    ax.set_title(f'{metric_name} Trend Analysis', fontsize=18, pad=20, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=15, labelpad=12)
    ax.set_ylabel(f'{metric_name} ({metric_unit})', fontsize=15, labelpad=12)

    # 设置坐标轴范围和刻度
    x_padding = max(1, max_x * 0.05)  # 扩展X轴范围
    ax.set_xlim(1 - x_padding, max_x + x_padding)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴只显示整数
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # 设置次要刻度

    # 设置y轴范围，留出适当空间
    y_min, y_max = ax.get_ylim()
    padding = (y_max - y_min) * 0.15  # 增加Y轴边距，避免图例遮挡
    ax.set_ylim(y_min - padding, y_max + padding)

    # 自定义网格线
    ax.grid(True, which='major', linestyle='-', alpha=0.5, color='#eeeeee')

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    # 设置坐标轴边框
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)

    # 优化图例（15组数据专用布局）
    if num_datasets > 10:
        # 超过10组数据时，将图例分为两列
        legend = ax.legend(fontsize=11, frameon=True, loc='upper left',
                           fancybox=True, framealpha=0.9, shadow=False,
                           edgecolor='#dddddd', ncol=2, columnspacing=1.5)
    else:
        legend = ax.legend(fontsize=11, frameon=True, loc='upper left',
                           fancybox=True, framealpha=0.9, shadow=False,
                           edgecolor='#dddddd')
    legend.get_frame().set_linewidth(0.5)
    plt.setp(legend.get_texts(), multialignment='left')  # 图例文本左对齐

    # 添加水平参考线（如果有意义）
    if metric in ['accuracy', 'precision', 'recall', 'f1']:
        ax.axhline(y=50, color='#cccccc', linestyle='--', linewidth=1, alpha=0.7)

    # 调整布局（增加底部边距，防止图例被截断）
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

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

    print(f"Found {len(json_files)} JSON files (supporting up to 15 datasets).")

    # 加载所有JSON文件
    data_dict = load_json_files(json_files)
    if not data_dict:
        print("Failed to load any data files.")
        return

    # 检查数据组数
    if len(data_dict) > 15:
        print(f"Warning: Detected {len(data_dict)} datasets, which exceeds the optimal 15. "
              "Consider reducing the number of datasets for better visualization.")

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