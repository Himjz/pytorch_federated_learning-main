from typing import List, Dict, Tuple, Optional, Union, Any

import torch

from preprocessing.utils import DataSplitter


class UniversalDataLoader(DataSplitter):
    """
    通用联邦学习数据加载器，继承自DataSplitter，
    提供完整的数据加载、处理和联邦划分功能
    """

    def __init__(self,
                 dataset_name: str = 'MNIST',
                 num_client: int = 1,
                 num_local_class: int = 10,
                 untrusted_strategies: Optional[List[float]] = None,
                 k: int = 1,
                 print_report: bool = True,
                 device: Optional[torch.device] = None,
                 preload_to_gpu: bool = False,
                 seed: int = 0,
                 root: str = './data',
                 download: bool = True,
                 cut: Optional[float] = None,
                 save: bool = False,
                 subset: Optional[Tuple[str, Union[int, float]]] = None,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 in_channels: Optional[int] = None,
                 augmentation: bool = False,
                 augmentation_params: Optional[Dict[str, Any]] = None):
        # 调用父类构造函数
        super().__init__(
            dataset_name=dataset_name,
            num_client=num_client,
            num_local_class=num_local_class,
            untrusted_strategies=untrusted_strategies,
            k=k,
            print_report=print_report,
            device=device,
            preload_to_gpu=preload_to_gpu,
            seed=seed,
            root=root,
            download=download,
            cut=cut,
            save=save,
            subset=subset,
            size=size,
            in_channels=in_channels,
            augmentation=augmentation,
            augmentation_params=augmentation_params
        )


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 示例：使用cut和subset参数
    loader = UniversalDataLoader(
        root="..",
        num_client=5,
        num_local_class=2,
        dataset_name='dt',
        seed=0,
        untrusted_strategies=[2.1, 3.5, 4.3, 0.5, 1.0],
        device=device,
        preload_to_gpu=False,
        in_channels=1,
        size=32,
        augmentation=True,
        cut=0.8,  # 保留80%的样本
        subset=('random', 3)  # 随机选择3个类别
    )

    loader.load()

    mean, std = loader.calculate_statistics()
    print(f"\n数据集统计信息: 均值={mean}, 标准差={std}")

    try:
        trainset_config, testset = loader.divide()

        print("\n数据集信息:")
        print(f"  类别数: {loader.num_classes}")
        print(f"  图像尺寸: {loader.image_size}x{loader.image_size}")
        print(f"  图像通道数: {loader.in_channels}")

        for i in range(3):
            client_id = f"f_0000{i}"
            if client_id in trainset_config['user_data']:
                trainset_config['user_data'][client_id].verify_loader()
            else:
                print(f"客户端 {client_id} 不存在")
    except Exception as e:
        print(f"划分数据集时出错: {e}")
