import json
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

with open('results/time.json','r',encoding='utf-8') as f:
    data = json.load(f)

local_train_time = data['local_train_time']
model_transfer_time = data['model_transfer_time']
global_agg_time = data['global_agg_time']
model_update_time = data['model_update_time']

# 创建一个2x2的子图布局，调整子图之间的间距
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图表1：本地训练时间
axes[0, 0].plot(local_train_time, 'b-', linewidth=1.5)
axes[0, 0].set_title('本地训练时间', fontsize=14)
axes[0, 0].set_xlabel('数据点序号', fontsize=12)
axes[0, 0].set_ylabel('时间 (秒)', fontsize=12)
axes[0, 0].grid(True, linestyle='--', alpha=0.7)

# 图表2：模型传输时间
axes[0, 1].plot(model_transfer_time, 'g-', linewidth=1.5)
axes[0, 1].set_title('模型传输时间', fontsize=14)
axes[0, 1].set_xlabel('数据点序号', fontsize=12)
axes[0, 1].set_ylabel('时间 (秒)', fontsize=12)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)

# 图表3：全局聚合时间
axes[1, 0].plot(global_agg_time, 'r-', linewidth=1.5)
axes[1, 0].set_title('全局聚合时间', fontsize=14)
axes[1, 0].set_xlabel('数据点序号', fontsize=12)
axes[1, 0].set_ylabel('时间 (秒)', fontsize=12)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)

# 图表4：模型更新时间
axes[1, 1].plot(model_update_time, 'm-', linewidth=1.5)
axes[1, 1].set_title('模型更新时间', fontsize=14)
axes[1, 1].set_xlabel('数据点序号', fontsize=12)
axes[1, 1].set_ylabel('时间 (秒)', fontsize=12)
axes[1, 1].grid(True, linestyle='--', alpha=0.7)

# 调整子图之间的间距
plt.subplots_adjust(
    left=0.1,    # 左侧边距
    right=0.95,  # 右侧边距
    bottom=0.08, # 底部边距
    top=0.92,    # 顶部边距
    wspace=0.25, # 子图之间的水平间距
    hspace=0.3   # 子图之间的垂直间距
)

# 添加主标题
plt.suptitle('联邦学习各阶段时间消耗分析', fontsize=18, fontweight='bold', y=0.96)

# 显示图形
plt.show()