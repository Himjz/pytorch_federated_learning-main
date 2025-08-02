import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime

# 错误日志文件路径
ERROR_LOG_FILE = "install_errors.log"

def log_error(message):
    """将错误信息写入日志文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] 错误: {message}\n")
            # 添加堆栈跟踪信息
            traceback_info = traceback.format_exc()
            if traceback_info.strip() != "NoneType: None":  # 避免空跟踪信息
                f.write(f"堆栈跟踪:\n{traceback_info}\n")
            f.write("-" * 80 + "\n")
    except Exception as e:
        print(f"记录错误日志失败: {e}")
        print(f"错误内容: {message}")

# 强制要求 Python 3.9+
if sys.version_info < (3, 9):
    error_msg = f"""
    此项目需要 Python 3.9 或更高版本，但您当前使用的是 Python {sys.version_info.major}.{sys.version_info.minor}.
    请升级 Python 到 3.9 或更高版本后再安装。
    """
    log_error(error_msg)
    sys.exit(error_msg)





class BasicProgress:
    """基础进度显示类，不依赖任何第三方库"""

    def __init__(self, total, description, unit='项'):
        self.total = total
        self.description = description
        self.unit = unit
        self.completed = 0
        self.start_time = time.time()
        self.last_updated = 0
        self._print_progress()

    def update(self, n=1):
        """更新进度"""
        self.completed = min(self.completed + n, self.total)
        # 控制更新频率，避免输出太频繁
        current_time = time.time()
        if self.completed == self.total or current_time - self.last_updated > 0.5:
            self._print_progress()
            self.last_updated = current_time

    def close(self):
        """完成进度显示"""
        if self.completed < self.total:
            self.completed = self.total
            self._print_progress()
        elapsed = time.time() - self.start_time
        print(f"\n{self.description} 完成，耗时 {elapsed:.1f} 秒")

    def _print_progress(self):
        """打印进度信息"""
        percentage = (self.completed / self.total) * 100 if self.total > 0 else 100
        # 简单的进度条，由#和空格组成
        bar_length = 30
        filled_length = int(bar_length * self.completed // self.total)
        bar = '#' * filled_length + ' ' * (bar_length - filled_length)
        print(f"\r{self.description}: [{bar}] {self.completed}/{self.total} {self.unit} ({percentage:.1f}%)", end='',
              flush=True)


def get_python_version():
    """获取 Python 版本信息"""
    return platform.python_version()


def get_cuda_info():
    """检查 CUDA 可用性并获取驱动版本"""
    cuda_available = False
    cuda_version = None

    # 定义版本提取正则表达式
    version_pattern = re.compile(r'(\d+\.\d+)')

    # 尝试通过 nvcc 获取 CUDA 编译器版本
    try:
        nvcc_output = subprocess.check_output(
            ['nvcc', '--version'],
            stderr=subprocess.STDOUT,
            timeout=5
        ).decode()

        match = version_pattern.search(nvcc_output)
        if match:
            cuda_available = True
            cuda_version = match.group(1)
            return cuda_available, cuda_version
    except Exception as e:
        log_error(f"通过 nvcc 获取 CUDA 版本失败: {e}")

    # 尝试通过 nvidia-smi 获取 CUDA 驱动版本
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi'],
            stderr=subprocess.STDOUT,
            timeout=5
        ).decode()

        for line in smi_output.split('\n'):
            if 'CUDA Version' in line:
                match = version_pattern.search(line)
                if match:
                    cuda_available = True
                    cuda_version = match.group(1)
                    return cuda_available, cuda_version
    except Exception as e:
        log_error(f"通过 nvidia-smi 获取 CUDA 版本失败: {e}")

    # 尝试通过 PyTorch 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            cuda_version = torch.version.cuda
            return cuda_available, cuda_version
    except ImportError|NameError:
        pass  # PyTorch 未安装是正常情况，无需记录错误
    except Exception as e:
        log_error(f"通过 PyTorch 检查 CUDA 失败: {e}")

    return cuda_available, cuda_version


def get_pytorch_source():
    """获取 PyTorch 专用源（根据 CUDA 环境）"""
    cuda_available, cuda_version = get_cuda_info()

    # 下载源设置
    default_index = "https://pypi.tuna.tsinghua.edu.cn/simple"
    torch_mirror_base = "https://mirrors.nju.edu.cn/pytorch/whl"

    if cuda_available and cuda_version:
        try:
            cuda_float = float(cuda_version)

            # 定义版本映射表
            version_mapping = [
                (12.8, "cu128"),
                (12.6, "cu126"),
                (12.4, "cu124"),
                (11.8, "cu118"),
            ]

            cuda_folder = None
            for min_version, folder in version_mapping:
                if cuda_float >= min_version:
                    cuda_folder = folder
                    break

            if cuda_folder:
                return f"{torch_mirror_base}/{cuda_folder}"
        except Exception as e:
            log_error(f"CUDA 版本解析失败 ({cuda_version}): {e}")

    return default_index


def count_requirements(file_path):
    """统计需求文件中的包数量"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        count = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                count += 1
        return count
    except Exception as e:
        error_msg = f"统计需求文件 {file_path} 失败: {e}"
        log_error(error_msg)
        print(error_msg)
        return 0


def install_with_progress(command, description, total_packages):
    """执行安装命令并显示进度"""
    print(f"\n{description}...")

    try:
        progress = BasicProgress(total_packages, description, '包')

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        installed_count = 0
        last_installed = None

        for line in process.stdout:
            if 'Installing collected packages:' in line:
                continue

            install_match = re.search(r'Installing (.*?)(\s|$)', line)
            if install_match and install_match.group(1) != last_installed:
                last_installed = install_match.group(1)
                installed_count = min(installed_count + 1, total_packages)
                progress.update(1)

        process.wait()
        progress.close()

        if process.returncode == 0:
            print(f"{description}成功")
            return True
        else:
            error_msg = f"{description}失败，返回代码: {process.returncode}"
            log_error(error_msg)
            print(error_msg)
            return False

    except Exception as e:
        error_msg = f"{description}过程中出错: {e}"
        log_error(error_msg)
        print(error_msg)
        return False


def run_command(command, description):
    """执行简单命令并处理错误"""
    print(f"\n{description}...")

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(f"{description}成功")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"{description}失败: {e.stdout}"
        log_error(error_msg)
        print(f"{description}失败，请查看错误日志获取详情")
        return False
    except Exception as e:
        error_msg = f"{description}过程中出错: {e}"
        log_error(error_msg)
        print(f"{description}失败，请查看错误日志获取详情")
        return False


def main():
    try:
        # 打印简要环境信息
        python_version = get_python_version()
        cuda_available, cuda_version = get_cuda_info()

        print("环境信息:")
        print(f"Python 版本: {python_version}")
        print(f"CUDA 可用性: {'可用' if cuda_available else '不可用'}")
        if cuda_available:
            print(f"CUDA 版本: {cuda_version}")

        # 1. 更新 pip
        if not run_command(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                "更新 pip"
        ):
            print("pip 更新失败，继续尝试安装依赖...")

        # 2. 安装基础依赖
        req_count = count_requirements("requirements/requirements.txt")
        if req_count > 0:
            if not install_with_progress(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements/requirements.txt"],
                    "安装基础依赖",
                    req_count
            ):
                print("基础依赖安装失败，可能导致后续步骤出错")
        else:
            print("未找到基础依赖需求文件或文件为空")

        # 3. 获取 PyTorch 源并安装相关依赖
        pytorch_source = get_pytorch_source()
        torch_req_count = count_requirements("requirements/requirements_torch.txt")
        if torch_req_count > 0:
            install_with_progress(
                [sys.executable, "-m", "pip", "install", "-r", "requirements/requirements_torch.txt", "-i",
                 pytorch_source],
                "安装 PyTorch 相关依赖",
                torch_req_count
            )
        else:
            print("未找到 PyTorch 依赖需求文件或文件为空")

        print("\n所有安装步骤执行完毕")
        print(f"如果遇到问题，请查看错误日志: {ERROR_LOG_FILE}")

    except Exception as e:
        error_msg = f"安装过程中发生致命错误: {e}"
        log_error(error_msg)
        print(f"\n安装失败: {error_msg}")
        print(f"详细错误信息已记录到: {ERROR_LOG_FILE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
