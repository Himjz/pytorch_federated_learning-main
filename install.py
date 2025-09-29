# 首先解决Windows编码问题
import sys
import shutil
import glob

if sys.platform.startswith('win32'):
    import codecs

    # 注册UTF-8编码处理，避免中文解码错误
    codecs.register(lambda name: codecs.lookup('utf-8') if name == 'cp65001' else None)
    # 重定向标准输出为UTF-8编码
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

import logging
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional

# 程序名称（用于日志目录命名）
PROGRAM_NAME = "pytorch_federated_learning"
# 日志文件路径变量（全局存储，用于最后打印）
LOG_FILE_PATH = None


def clean_invalid_distributions() -> None:
    """
    清理site-packages中名称带有波浪号的无效分布文件夹（如~umpy）
    这些通常是pip安装过程中残留的临时文件
    """
    try:
        # 获取site-packages目录路径
        site_packages = Path(sys.prefix) / "Lib" / "site-packages"

        if not site_packages.exists():
            logging.warning(f"未找到site-packages目录: {site_packages}")
            return

        # 查找所有名称以波浪号开头的文件夹
        invalid_pattern = str(site_packages / "~*")
        invalid_folders = glob.glob(invalid_pattern)

        if invalid_folders:
            print(f"\n发现{len(invalid_folders)}个无效的分布文件夹，正在清理...")
            logging.info(f"发现无效分布文件夹: {invalid_folders}")

            for folder in invalid_folders:
                try:
                    # 尝试删除文件夹
                    if os.path.isdir(folder):
                        shutil.rmtree(folder)
                    else:
                        os.remove(folder)
                    print(f"已清理无效文件夹: {os.path.basename(folder)}")
                    logging.info(f"已清理无效文件夹: {folder}")
                except Exception as e:
                    print(f"清理文件夹 {os.path.basename(folder)} 失败: {str(e)}")
                    logging.warning(f"清理文件夹 {folder} 失败: {str(e)}")
        else:
            logging.info("未发现无效的分布文件夹")

    except Exception as e:
        print(f"清理无效分布文件夹时发生错误: {str(e)}")
        logging.error(f"清理无效分布文件夹时发生错误: {str(e)}")


def check_python_version() -> None:
    """检查Python版本是否满足要求（3.9+），不满足则退出程序"""
    required_version = (3, 9)
    current_version = sys.version_info[:2]

    if current_version < required_version:
        error_msg = (
            f"此项目需要 Python {required_version[0]}.{required_version[1]} 或更高版本，"
            f"但您当前使用的是 Python {current_version[0]}.{current_version[1]}.\n"
            "请升级 Python 到 3.9 或更高版本后再安装。"
        )
        sys.exit(error_msg)


def get_python_version() -> str:
    """获取当前Python版本信息"""
    return platform.python_version()


def get_cuda_info() -> Tuple[bool, Optional[str]]:
    """
    检查CUDA可用性并获取版本信息
    返回: Tuple[bool, Optional[str]]: (CUDA是否可用, CUDA版本号或None)
    """
    cuda_available = False
    cuda_version = None
    version_pattern = re.compile(r'(\d+\.\d+)')  # 用于提取版本号的正则表达式

    # 尝试通过nvcc获取CUDA编译器版本（优先方法）
    try:
        nvcc_output = subprocess.check_output(
            ['nvcc', '--version'],
            stderr=subprocess.STDOUT,
            timeout=5,
            text=True
        )

        match = version_pattern.search(nvcc_output)
        if match:
            cuda_available = True
            cuda_version = match.group(1)
            logging.info(f"通过 nvcc 检测到 CUDA 版本: {cuda_version}")
            print(f"通过 nvcc 检测到 CUDA 版本: {cuda_version}")
            return cuda_available, cuda_version

    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("未找到nvcc，无法通过nvcc检测CUDA版本")
        print("未找到nvcc，尝试其他方式检测CUDA版本")
    except TimeoutError:
        logging.warning("nvcc版本检测超时")
        print("nvcc检测超时，尝试其他方式检测CUDA版本")
    except Exception as e:
        logging.error(f"通过nvcc检测CUDA时发生意外错误: {str(e)}")
        print("检测CUDA时发生错误，尝试其他方式")

    # 尝试通过nvidia-smi获取CUDA驱动版本
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi'],
            stderr=subprocess.STDOUT,
            timeout=5,
            text=True
        )

        for line in smi_output.split('\n'):
            if 'CUDA Version' in line:
                match = version_pattern.search(line)
                if match:
                    cuda_available = True
                    cuda_version = match.group(1)
                    logging.info(f"通过 nvidia-smi 检测到 CUDA 版本: {cuda_version}")
                    print(f"通过 nvidia-smi 检测到 CUDA 版本: {cuda_version}")
                    return cuda_available, cuda_version

    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("未找到nvidia-smi，无法通过nvidia-smi检测CUDA版本")
        print("未找到nvidia-smi，尝试其他方式检测CUDA版本")
    except TimeoutError:
        logging.warning("nvidia-smi版本检测超时")
        print("nvidia-smi检测超时，尝试其他方式检测CUDA版本")
    except Exception as e:
        logging.error(f"通过nvidia-smi检测CUDA时发生意外错误: {str(e)}")
        print("检测CUDA时发生错误，尝试其他方式")

    # 尝试通过PyTorch检查CUDA（如果已安装）
    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            cuda_version = torch.version.cuda
            logging.info(f"通过 PyTorch 检测到 CUDA 版本: {cuda_version}")
            print(f"通过 PyTorch 检测到 CUDA 版本: {cuda_version}")
            return cuda_available, cuda_version

    except (ImportError, NameError):
        logging.info("PyTorch未安装，跳过CUDA检查")
        print("PyTorch未安装，将在后续步骤安装")
    except Exception as e:
        logging.error(f"通过PyTorch检测CUDA时发生错误: {str(e)}")
        print("检测CUDA时发生错误")

    logging.info("未检测到CUDA环境，将安装CPU版本依赖")
    print("未检测到CUDA环境，将安装CPU版本依赖")
    return cuda_available, cuda_version


def generate_environment_report() -> str:
    """生成并返回环境配置报告"""
    python_version = get_python_version()
    cuda_available, cuda_version = get_cuda_info()

    report = (
        "==========================\n"
        "环境配置报告\n"
        "==========================\n"
        f"Python 版本: {python_version}\n"
        f"CUDA 可用性: {'可用' if cuda_available else '不可用'}\n"
        f"CUDA 版本: {cuda_version if cuda_version else '未找到'}"
    )
    logging.info(f"生成环境报告:\n{report}")
    return report


def get_sources() -> Tuple[List[str], List[str]]:
    """
    获取分开管理的源列表
    返回: Tuple[List[str], List[str]]:
        第一个列表是PyTorch专用源（官方源优先+国内源备用）
        第二个列表是基础包源（国内源优先+官方源备用）
    """
    # 基础包源配置（国内源优先+官方源备用）
    base_packages_primary = "https://pypi.tuna.tsinghua.edu.cn/simple"
    base_packages_backup = "https://pypi.org/simple"
    base_sources = [base_packages_primary, base_packages_backup]

    # PyTorch专用源配置（官方源优先+国内源备用）
    torch_primary_base = "https://download.pytorch.org/whl"  # 官方源优先
    torch_backup_base = "https://mirrors.nju.edu.cn/pytorch/whl"  # 国内源备用
    torch_sources = [torch_primary_base, torch_backup_base]

    # 记录源信息到日志（不输出到控制台）
    logging.info(f"基础包源列表: {base_sources}")
    logging.info(f"PyTorch源列表（基础）: {torch_sources}")

    # 根据CUDA版本调整PyTorch源：支持CUDA 12.9及以下版本
    cuda_available, cuda_version = get_cuda_info()
    if cuda_available and cuda_version:
        try:
            cuda_float = float(cuda_version)
            # 版本映射，优先级从高到低
            version_mapping = [
                (12.9, "cu129"),
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
                # 为每个PyTorch源添加CUDA路径
                configured_torch_sources = [
                    f"{source}/{cuda_folder}" for source in torch_sources
                ]
                logging.info(f"配置PyTorch源（CUDA {cuda_folder}，兼容{cuda_version}）: {configured_torch_sources}")
                print(f"配置PyTorch源（CUDA {cuda_folder}，兼容当前检测到的{cuda_version}）")
                return configured_torch_sources, base_sources
            else:
                logging.warning(f"CUDA {cuda_version} 版本较旧，使用CPU版本PyTorch源")
                print(f"CUDA {cuda_version} 版本较旧，将使用CPU版本PyTorch")

        except ValueError:
            logging.error(f"无法解析CUDA版本: {cuda_version}，使用默认PyTorch源")
            print(f"CUDA版本解析失败，将使用CPU版本PyTorch")

    # 如果没有可用CUDA，使用CPU版本源
    cpu_torch_sources = [f"{source}/cpu" for source in torch_sources]
    logging.info(f"使用CPU版本PyTorch源: {cpu_torch_sources}")
    return cpu_torch_sources, base_sources


def setup_logging() -> None:
    """配置日志系统，将日志写入系统缓存目录"""
    global LOG_FILE_PATH  # 声明使用全局变量

    # 获取系统缓存目录
    if sys.platform.startswith('win32'):
        # Windows系统通常使用LOCALAPPDATA
        cache_dir = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.cache'))
    elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        # Linux和macOS使用~/.cache
        cache_dir = Path.home() / '.cache'
    else:
        # 其他系统默认使用用户目录下的.cache
        cache_dir = Path.home() / '.cache'

    # 创建程序日志目录
    log_dir = cache_dir / PROGRAM_NAME / 'install'
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件路径
    LOG_FILE_PATH = log_dir / 'install.log'

    # 配置日志 - 增加DEBUG级别以捕获更多细节
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'),
            # 不添加StreamHandler，避免日志输出到控制台
        ]
    )

    logging.info("日志系统初始化完成")
    logging.info(f"日志文件路径: {LOG_FILE_PATH}")


def parse_package_installation(output: str, source: str) -> None:
    """
    解析安装输出，提取并记录包的下载信息
    参数:
        output: 命令输出内容
        source: 下载源
    """
    # 记录完整输出
    logging.debug(f"安装命令完整输出:\n{output}")

    # 提取已安装/已下载的包信息
    lines = output.split('\n')
    for line in lines:
        line = line.strip()
        # 匹配安装/升级包的行
        if line.startswith(('Installing ', 'Upgrading ', 'Downloading ')):
            logging.info(f"包操作: {line} (源: {source})")
        # 匹配已满足依赖的行
        elif line.startswith('Requirement already satisfied:'):
            pkg_info = line.replace('Requirement already satisfied:', '').strip()
            logging.info(f"包已安装: {pkg_info}")
        # 匹配成功安装的行
        elif line.startswith('Successfully installed'):
            pkgs = line.replace('Successfully installed', '').strip()
            logging.info(f"安装成功: {pkgs} (源: {source})")


def run_command_with_retry(command_base: List[str], sources: List[str], description: str) -> bool:
    """
    带源重试机制的命令执行函数：当主源失败时，自动尝试其他源
    增强日志记录，详细记录所有包的下载信息并实时显示进度
    参数:
        command_base: 基础命令列表（不含 -i 源参数）
        sources: 源列表（按优先级排序）
        description: 命令描述
    返回: bool: 是否成功执行
    """
    # 添加进度显示参数
    command_base = command_base + ["--progress-bar", "on"]
    output = ''
    for i, source in enumerate(sources):
        try:
            # 构建完整命令（添加当前源参数）
            command = command_base + ["-i", source]
            logging.info(f"执行命令（源 {i + 1}/{len(sources)}）: {' '.join(command)}")
            print(f"\n正在{description}（尝试 {i + 1}/{len(sources)}）...")
            print(f"使用源: {source}")

            # 创建进程，实时捕获并显示输出
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # 行缓冲
            )

            # 实时输出和记录
            output_lines = []
            if process.stdout:
                for line in process.stdout:
                    try:
                        # 尝试直接处理行（已设置UTF-8编码）
                        line = line.rstrip()
                        # 打印到控制台显示进度
                        print(line)
                        # 保存到列表用于后续日志记录
                        output_lines.append(line)
                    except UnicodeDecodeError as e:
                        # 编码错误处理
                        logging.warning(
                            f"解码输出时发生错误: {str(e)}, 行内容: {line.encode('utf-8', errors='replace')}")
                        # 使用替换错误的方式处理
                        line = line.encode('utf-8', errors='replace').decode('utf-8')
                        print(line)
                        output_lines.append(line)

            # 等待进程完成
            process.wait()

            # 检查返回码
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

            # 合并输出内容
            output = "\n".join(output_lines)

            # 解析并记录包安装信息
            parse_package_installation(output, source)

            logging.info(f"{description}成功（使用源: {source}）")
            print(f"\n{description}成功")
            return True

        except subprocess.CalledProcessError as e:
            # 错误输出也记录包相关信息
            logging.error(f"使用源 {source} {description}失败: 返回代码 {e.returncode}")
            logging.error(f"错误输出内容:\n{output if 'output' in locals() else '无输出'}")
            if 'output' in locals():
                parse_package_installation(output, source)  # 即使失败也记录已安装的包
            print(f"\n{description}尝试 {i + 1} 失败")
            if i < len(sources) - 1:
                print(f"正在尝试下一个源...")
            else:
                print(f"\n{description}失败")
        except Exception as e:
            logging.error(f"使用源 {source} 执行命令时发生错误: {str(e)}")
            print(f"\n{description}尝试 {i + 1} 发生错误: {str(e)}")
            if i < len(sources) - 1:
                print(f"正在尝试下一个源...")
            else:
                print(f"\n{description}失败")

    return False


def main() -> None:
    """主函数，协调环境检测和依赖安装过程"""
    # 初始化日志
    setup_logging()

    # 清理无效的分布文件夹（如~umpy）
    clean_invalid_distributions()

    # 检查Python版本
    check_python_version()

    # 生成并打印环境报告
    report = generate_environment_report()
    print(report)

    # 获取分开管理的源列表
    torch_sources, base_sources = get_sources()

    # 升级pip（使用基础包源）
    pip_upgrade_base = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    if not run_command_with_retry(pip_upgrade_base, base_sources, "升级pip"):
        logging.warning("pip升级失败")
        print("警告：pip升级失败，继续安装依赖但可能会遇到问题")

    # 安装基础依赖（使用基础包源）
    base_deps_success = run_command_with_retry(
        [sys.executable, "-m", "pip", "install", "-r", "requirements/requirements_base.txt"],
        base_sources,
        "安装基础依赖"
    )
    if not base_deps_success:
        logging.error("基础依赖安装失败，所有源均尝试过")
        # 失败时也打印日志路径
        print(f"\n安装日志已保存至: {LOG_FILE_PATH}")
        sys.exit("基础依赖安装失败，无法继续")

    # 安装PyTorch相关依赖（使用PyTorch专用源）
    torch_deps_success = run_command_with_retry(
        [sys.executable, "-m", "pip", "install", "-r", "requirements/requirements_torch.txt"],
        torch_sources,
        "安装PyTorch相关依赖"
    )
    if not torch_deps_success:
        logging.error("PyTorch相关依赖安装失败，所有源均尝试过")
        # 失败时也打印日志路径
        print(f"\n安装日志已保存至: {LOG_FILE_PATH}")
        sys.exit("PyTorch相关依赖安装失败，无法继续")

    # 安装完成后再次清理可能产生的无效文件夹
    clean_invalid_distributions()

    logging.info("所有依赖安装完成")
    print("\n所有依赖安装完成！")
    # 成功完成后打印日志路径
    print(f"安装日志已保存至: {LOG_FILE_PATH}")
    print("如需查看详细安装过程或排查问题，可查看此日志文件")


if __name__ == "__main__":
    main()
