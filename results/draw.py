import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.signal import savgol_filter

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid.axis"] = "y"
plt.rcParams["grid.color"] = "#eeeeee"
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["legend.loc"] = "upper left"

# 字体设置
plt.rcParams["font.size"] = 12
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

# 指标映射
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

# 配色方案
PROFESSIONAL_COLORS = [
    '#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974',
    '#64B5CD', '#4878CF', '#6ACC65', '#D65F5F', '#B47CC7'
]

# 算法样式（线型+标记）
ALGORITHM_STYLES = {
    'FedAvg': ('-', 'o'),  # 实线+圆点
    'FedShapley': ('--', 's'),  # 虚线+方点
}

# 信任度颜色深浅
TRUST_LEVELS = {
    'FullTrusted': 1.0,  # 最深
    'High': 0.8,  # 较深
    'Low': 0.6,  # 较浅
}


def load_json_files(file_paths):
    data_dict = {}
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            file_name = os.path.splitext(os.path.basename(path))[0]
            data_dict[file_name] = data
            print(f"Successfully loaded file: {file_name}")
        except Exception as e:
            print(f"Failed to load file {path}: {str(e)}")
    return data_dict


def get_common_metrics(data_dict):
    if not data_dict:
        return []
    first_data = next(iter(data_dict.values()))
    common_metrics = set(first_data.keys())
    for data in data_dict.values():
        common_metrics.intersection_update(data.keys())

    priority_order = ['accuracy', 'precision', 'recall', 'f1', 'loss',
                      'client_train_avg', 'server_train', 'client_update_avg',
                      'global_agg', 'model_transfer_avg']
    sorted_metrics = []
    for metric in priority_order:
        if metric in common_metrics:
            sorted_metrics.append(metric)
            common_metrics.remove(metric)
    sorted_metrics.extend(sorted(common_metrics))
    return sorted_metrics


def get_metric_display_name(metric):
    return METRIC_NAMES.get(metric, metric.replace('_', ' ').title())


def get_metric_unit(metric):
    return METRIC_UNITS.get(metric, '')


def smooth_curve(y, window_size=5, polyorder=3):
    if len(y) < window_size:
        return y
    return savgol_filter(y, window_length=window_size, polyorder=polyorder)


def parse_filename(filename):
    """解析文件名，返回(信任度, 算法)"""
    for trust in TRUST_LEVELS:
        if filename.startswith(trust):
            algorithm = filename[len(trust) + 1:]
            return trust, algorithm
    for alg in ALGORITHM_STYLES:
        if filename.endswith(alg):
            trust = filename[:-len(alg) - 1]
            return trust, alg
    return None, None


def plot_metric(data_dict, metric, use_meta_info=False, figsize=(12, 7),
                output_dir='metric_plots', show=False, vector_formats=['svg', 'pdf'], dpi=300):
    """
    绘制指标对比图

    参数:
        data_dict: 包含数据的字典
        metric: 要绘制的指标
        use_meta_info: 是否使用文件元信息(信任度和算法)进行分组对比
        figsize: 图表大小
        output_dir: 输出目录
        show: 是否显示图表
        vector_formats: 矢量图格式列表
        dpi: 图像分辨率
    """
    fig, ax = plt.subplots(figsize=figsize)
    metric_name = get_metric_display_name(metric)
    metric_unit = get_metric_unit(metric)
    max_x = 0

    if use_meta_info:
        # 按信任度和算法分组
        groups = {}
        for filename in data_dict:
            trust, algorithm = parse_filename(filename)
            if trust not in groups:
                groups[trust] = {}
            groups[trust][algorithm] = data_dict[filename]

        # 为每个信任度分配基础颜色
        trust_list = sorted(groups.keys(), key=lambda x: TRUST_LEVELS[x], reverse=True)
        color_map = {trust: PROFESSIONAL_COLORS[i] for i, trust in enumerate(trust_list)}

        # 绘制所有曲线
        for trust in trust_list:
            algorithms = groups[trust]
            base_color = color_map[trust]
            # 调整颜色深浅
            color = mcolors.to_rgb(base_color)
            color = tuple(c * TRUST_LEVELS[trust] for c in color)

            for algorithm in algorithms:
                if metric not in algorithms[algorithm]:
                    continue
                values = np.array(algorithms[algorithm][metric])
                x = np.arange(1, len(values) + 1)
                max_x = max(max_x, len(x))

                # 平滑曲线
                smooth_vals = smooth_curve(values)

                # 获取线型和标记
                line_style, marker = ALGORITHM_STYLES.get(algorithm, ('-', 'o'))

                # 绘制曲线
                ax.plot(
                    x, smooth_vals,
                    label=f'{trust}-{algorithm}',
                    color=color,
                    linestyle=line_style,
                    linewidth=2,
                    marker=marker,
                    markersize=4,
                    markevery=max(1, len(x) // 20),
                    alpha=0.9
                )
    else:
        # 不使用元信息，直接按文件名绘制
        color_index = 0
        for filename, data in data_dict.items():
            if metric not in data:
                continue

            values = np.array(data[metric])
            x = np.arange(1, len(values) + 1)
            max_x = max(max_x, len(x))

            # 平滑曲线
            smooth_vals = smooth_curve(values)

            # 获取颜色和样式
            color = PROFESSIONAL_COLORS[color_index % len(PROFESSIONAL_COLORS)]
            line_style, marker = ALGORITHM_STYLES[list(ALGORITHM_STYLES.keys())[color_index % len(ALGORITHM_STYLES)]]

            # 绘制曲线
            ax.plot(
                x, smooth_vals,
                label=filename,
                color=color,
                linestyle=line_style,
                linewidth=2,
                marker=marker,
                markersize=4,
                markevery=max(1, len(x) // 20),
                alpha=0.9
            )

            color_index += 1

    # 坐标轴设置
    ax.set_title(f'{metric_name} Comparison', fontsize=16, pad=15)
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel(f'{metric_name} ({metric_unit})', fontsize=14)
    ax.set_xlim(0, max_x + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # 图例设置
    ax.legend(fontsize=10, loc='best')

    # 保存图表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{metric}.png')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved PNG: {save_path}")

        for fmt in vector_formats:
            vec_path = os.path.join(output_dir, f'{metric}.{fmt}')
            fig.savefig(vec_path, bbox_inches='tight')
            print(f"Saved {fmt}: {vec_path}")

    if show:
        plt.show()
    return fig


def main():
    # 配置参数 - 在这里修改参数值
    use_meta_info = False  # 是否使用文件元信息进行对比，默认关闭
    show_plots = False  # 是否显示图表
    output_dir = 'metric_plots'  # 图表输出目录

    json_files = [f for f in os.listdir('') if f.endswith('.json')]
    if not json_files:
        print("No JSON files found.")
        return

    print(f"Found {len(json_files)} JSON files.")
    data_dict = load_json_files(json_files)
    if not data_dict:
        print("No data loaded.")
        return

    metrics = get_common_metrics(data_dict)
    print(f"Metrics to plot: {', '.join(metrics)}")

    for metric in metrics:
        try:
            plot_metric(
                data_dict,
                metric,
                use_meta_info=use_meta_info,
                output_dir=output_dir,
                show=show_plots
            )
        except Exception as e:
            print(f"Error plotting {metric}: {str(e)}")

    print("Plotting completed.")


if __name__ == "__main__":
    main()
