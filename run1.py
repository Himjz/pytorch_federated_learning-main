# 导入 subprocess 模块，用于在 Python 脚本里执行外部命令
import subprocess

# 定义要执行的命令，以列表形式存储，每个元素为命令的一部分
# 此命令用于执行 Python 脚本 fl_main.py 并指定配置文件路径
cmd = ['python', 'main3.py', '--config', './config/test_config.yaml']

# 调用 subprocess.run 函数执行命令
# 不设置 capture_output 参数，命令的输出会直接显示在终端
# 也不设置 text 参数，让输出保持默认模式
subprocess.run(cmd)