import platform
import re
import subprocess
import sys

from setuptools import setup, find_packages

# 强制要求 Python 3.9+
if sys.version_info < (3, 9):
    error_msg = f"""
    此项目需要 Python 3.9 或更高版本，但您当前使用的是 Python {sys.version_info.major}.{sys.version_info.minor}.
    请升级 Python 到 3.9 或更高版本后再安装。
    """
    sys.exit(error_msg)


def get_python_version():
    """获取 Python 版本信息"""
    return platform.python_version()


def get_cuda_info():
    """检查 CUDA 可用性并获取驱动版本，增强健壮性"""
    cuda_available = False
    cuda_version = None

    # 定义版本提取正则表达式
    version_pattern = re.compile(r'(\d+\.\d+)')

    # 尝试通过 nvcc 获取 CUDA 编译器版本（优先方法）
    try:
        nvcc_output = subprocess.check_output(
            ['nvcc', '--version'],
            stderr=subprocess.STDOUT,
            timeout=5
        ).decode()

        # 使用正则表达式提取版本号
        match = version_pattern.search(nvcc_output)
        if match:
            cuda_available = True
            cuda_version = match.group(1)
            print(f"通过 nvcc 检测到 CUDA 版本: {cuda_version}")
            return cuda_available, cuda_version
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
        print(f"无法通过 nvcc 获取 CUDA 版本: {e}")

    # 尝试通过 nvidia-smi 获取 CUDA 驱动版本
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi'],
            stderr=subprocess.STDOUT,
            timeout=5
        ).decode()

        # 查找 CUDA Version 字段
        for line in smi_output.split('\n'):
            if 'CUDA Version' in line:
                match = version_pattern.search(line)
                if match:
                    cuda_available = True
                    cuda_version = match.group(1)
                    print(f"通过 nvidia-smi 检测到 CUDA 版本: {cuda_version}")
                    return cuda_available, cuda_version
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError) as e:
        print(f"无法通过 nvidia-smi 获取 CUDA 版本: {e}")

    # 尝试通过 PyTorch 检查 CUDA（如果已安装）
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            cuda_version = torch.version.cuda
            print(f"通过 PyTorch 检测到 CUDA 版本: {cuda_version}")
            return cuda_available, cuda_version
    except ImportError | NameError:
        print("PyTorch 未安装，跳过 CUDA 检查")
    except Exception as e:
        print(f"检查 PyTorch CUDA 支持时出错: {e}")

    print("未检测到 CUDA 环境，将安装 CPU 版本依赖")
    return cuda_available, cuda_version


def generate_report():
    """生成环境报告"""
    python_version = get_python_version()
    cuda_available, cuda_version = get_cuda_info()

    r = f"""
    ==========================
    环境配置报告
    ==========================
    Python 版本: {python_version}
    CUDA 可用性: {'可用' if cuda_available else '不可用'}
    CUDA 版本: {cuda_version if cuda_version else '未找到'}
    """
    return r


def get_dependencies():
    """根据 CUDA 环境获取所有依赖项及下载源，优化版本映射逻辑"""
    cuda_available, cuda_version = get_cuda_info()

    # 基础依赖（非 PyTorch 相关）
    base_deps = [
        'numpy~=2.3.1',
        'scipy~=1.16.0',
        'Pillow~=11.3.0',
        'matplotlib~=3.10.1',
        'tqdm~=4.67.1',
        'opencv-python~=4.11.0.86',
        'scikit-learn~=1.7.0',
        'colorama~=0.4.6',
        'pykeops~=2.3',
        'pyyaml~=6.0.2',
        'setuptools>=61.0',
    ]

    # PyTorch 相关依赖
    torch_deps = [
        "torch~=2.7.1",
        "torchvision~=0.22.1",
        "torchaudio~=2.7.1"
    ]

    # 下载源设置
    default_index = "https://pypi.tuna.tsinghua.edu.cn/simple"  # 默认源（清华镜像）
    torch_mirror_base = "https://mirrors.nju.edu.cn/pytorch/whl"  # GPU版PyTorch专用源

    # 处理 PyTorch 源
    if cuda_available and cuda_version:
        print(f"检测到 CUDA {cuda_version}，将从 PyTorch 专用镜像安装 GPU 版本")

        # 将版本转换为浮点数进行比较
        try:
            cuda_float = float(cuda_version)

            # 定义版本映射表（CUDA版本 -> wheel目录）
            version_mapping = [
                (12.8, "cu128"),
                (12.6, "cu126"),
                (12.4, "cu124"),
                (11.8, "cu118"),
            ]

            # 查找匹配的 CUDA 目录
            cuda_folder = None
            for min_version, folder in version_mapping:
                if cuda_float >= min_version:
                    cuda_folder = folder
                    break

            if cuda_folder:
                print(f"选择 CUDA 版本对应目录: {cuda_folder}")
                dependency_links = [f"{torch_mirror_base}/{cuda_folder}"]
            else:
                print(f"警告：CUDA {cuda_version} 版本较旧，可能不受支持，将使用 CPU 版本")
                dependency_links = [default_index]

        except (ValueError, IndexError) as e:
            print(f"CUDA 版本解析失败 ({cuda_version}): {e}，将使用默认源安装 CPU 版本")
            dependency_links = [default_index]
    else:
        print("未检测到 CUDA，将从默认源安装 CPU 版本")
        dependency_links = [default_index]

    # 合并所有依赖
    all_deps = base_deps + torch_deps
    return all_deps, dependency_links


# 生成并打印报告
report = generate_report()
print(report)

# 获取依赖项和下载源
dependencies, links = get_dependencies()

# 设置项目信息
setup(
    name='environment_checker',
    version='1.0.0',
    description='环境配置检查工具',
    author='Your Name',
    author_email='your.email@example.com',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=dependencies,
    dependency_links=links,
    # 添加元数据描述依赖源
    long_description="""
    # Environment Checker

    自动检测系统环境并安装适配的深度学习依赖：
    - 自动识别 CUDA 版本并安装对应 GPU 版本 PyTorch
    - 非 PyTorch 依赖使用清华镜像源加速下载
    - 支持 Python 3.9+
    """,
    long_description_content_type='text/markdown',
)