import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet34_Weights, ResNet101_Weights, \
    ResNet152_Weights, MobileNet_V2_Weights

"""
我们提供了可能在 FedD3 实验中使用的模型，如下所示：
    - 为 CIFAR-10 定制的 AlexNet 模型（AlexCifarNet），包含 1756426 个参数
    - 为 MNIST 定制的 LeNet 模型，包含 61706 个参数
    - 更多的 ResNet 模型
    - 更多的 VGG 模型
    - MobileNet 模型
    - ShuffleNet 模型
"""


# 为 CIFAR-10 定制的 AlexNet 模型，包含 1756426 个参数
class AlexCifarNet(nn.Module):
    supported_dims = {32, 300}  # 支持 32x32 和 300x300 的输入尺寸

    def __init__(self, num_classes=10, in_channels=1, input_size=(300, 300)):
        super(AlexCifarNet, self).__init__()
        # 验证输入尺寸是否受支持
        if input_size[0] not in self.supported_dims or input_size[1] not in self.supported_dims:
            raise ValueError(f"输入尺寸 {input_size} 不受支持，支持的尺寸为 {self.supported_dims}")

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 提前计算全连接层的输入维度
        with torch.no_grad():
            test_input = torch.randn(1, in_channels, *input_size)
            test_output = self.features(test_input)
            test_output = test_output.view(test_output.size(0), -1)
            flatten_size = test_output.size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# 为 MNIST 定制的 LeNet 模型，包含 61706 个参数


class LeNet(nn.Module):
    supported_dims = {256}    # 导入农业数据集时将该参数改为300

    def __init__(self, num_classes=10, in_channels=1, input_size=(256, 256)):
        super(LeNet, self).__init__()
        # 验证输入尺寸是否受支持
        if input_size[0] not in self.supported_dims or input_size[1] not in self.supported_dims:
            raise ValueError(f"输入尺寸 {input_size} 不受支持，支持的尺寸为 {self.supported_dims}")
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 提前计算全连接层的输入维度
        with torch.no_grad():
            test_input = torch.randn(1, in_channels, *input_size)
            test_output = F.leaky_relu(self.conv1(test_input), inplace=True)
            test_output = self.pool(test_output)
            test_output = F.leaky_relu(self.conv2(test_output), inplace=True)
            test_output = self.pool(test_output)
            test_output = F.leaky_relu(self.conv3(test_output), inplace=True)
            test_output = self.pool(test_output)
            test_output = test_output.view(test_output.size(0), -1)
            flatten_size = test_output.size(1)
        self.fc1 = nn.Linear(flatten_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), inplace=True)
        out = self.pool(out)
        out = F.leaky_relu(self.conv2(out), inplace=True)
        out = self.pool(out)
        out = F.leaky_relu(self.conv3(out), inplace=True)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.fc1(out), inplace=True)
        out = self.dropout(out)
        out = F.leaky_relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out


# 更多的 ResNet 模型
def generate_resnet(num_classes=10, in_channels=1, model_name="ResNet18"):
    if model_name == "ResNet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "ResNet34":
        model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == "ResNet101":
        model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    elif model_name == "ResNet152":
        model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    return model


# 更多的 VGG 模型
def generate_vgg(num_classes=10, in_channels=1, model_name="vgg11"):
    """
    生成适配任意输入通道数的 VGG 模型

    参数:
    - num_classes: 分类类别数
    - in_channels: 输入图像通道数
    - model_name: 模型名称，支持: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
    """
    # 规范化模型名称并检查有效性
    model_name = model_name.lower().replace('_', '')
    valid_models = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    if not any(m in model_name for m in valid_models):
        raise ValueError(f"不支持的模型名称: {model_name}")

    # 获取模型创建函数和预训练标志
    use_pretrained = 'bn' in model_name
    create_fn = getattr(models, model_name)

    # 创建模型
    model = create_fn(weights='DEFAULT' if use_pretrained else None)

    # 修改输入通道
    if in_channels != 3:
        first_conv = model.features[0]
        new_conv = nn.Conv2d(
            in_channels, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        # 预训练模型的权重处理
        if use_pretrained and in_channels > 0:
            channels_to_copy = min(in_channels, 3)
            new_conv.weight.data[:, :channels_to_copy] = first_conv.weight.data[:, :channels_to_copy]

            # 随机初始化新增通道
            if in_channels < 3:
                new_conv.weight.data[:, channels_to_copy:].normal_(0, 0.01)

        model.features[0] = new_conv

    # 修改分类器输出
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    return model


class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=32, max_kernel_size=3):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.init_channels = in_channels
        self.max_kernel_size = max_kernel_size

        # 动态卷积块（将在forward中确定具体参数）
        self.conv_blocks = nn.ModuleList([
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),  # 初始卷积层
            nn.ReLU(),
            nn.BatchNorm2d(in_channels)
        ])

        # 中间卷积组（动态调整）
        self.mid_conv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels * 4)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应池化确保固定输出尺寸
            nn.Flatten(),
            nn.Linear(in_channels * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 计算初始特征图尺寸
        h, w = x.shape[2], x.shape[3]

        # 动态调整第一个卷积块的池化操作
        for layer in self.conv_blocks:
            x = layer(x)

        # 根据当前特征图尺寸确定池化参数
        current_h, current_w = x.shape[2], x.shape[3]
        pool_kernel = min(2, current_h // 2, current_w // 2)
        if pool_kernel >= 1:
            x = F.max_pool2d(x, kernel_size=pool_kernel, stride=pool_kernel)

        # 处理中间卷积组
        for i, layer in enumerate(self.mid_conv):
            x = layer(x)
            # 每两个卷积层后进行一次池化（动态调整）
            if (i + 1) % 3 == 0:  # 每经过一个完整的卷积+BN块后池化
                current_h, current_w = x.shape[2], x.shape[3]
                pool_kernel = min(2, current_h // 2, current_w // 2)
                if pool_kernel >= 1:
                    x = F.max_pool2d(x, kernel_size=pool_kernel, stride=pool_kernel)

        # 分类头
        x = self.classifier(x)
        return x


class EfficientCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, input_size=(256, 256)):
        super(EfficientCNN, self).__init__()
        # 验证输入尺寸
        if input_size != (256, 256):
            raise ValueError("仅支持 256x256 输入尺寸")

        # 第一组卷积块 - 使用深度可分离卷积
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128
        )

        # 第二组卷积块 - 使用深度可分离卷积
        self.group2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64
        )

        # 第三组卷积块 - 使用深度可分离卷积
        self.group3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32
        )

        # 第四组卷积块 - 使用深度可分离卷积
        self.group4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        )

        # 空间金字塔池化层 - 提取多尺度特征
        self.spp = nn.AdaptiveAvgPool2d((4, 4))  # 固定输出4x4

        # 动态计算全连接层的输入维度
        with torch.no_grad():
            test_input = torch.randn(1, in_channels, *input_size)
            test_output = self.group1(test_input)
            test_output = self.group2(test_output)
            test_output = self.group3(test_output)
            test_output = self.group4(test_output)
            test_output = self.spp(test_output)
            test_output = test_output.view(test_output.size(0), -1)
            flatten_size = test_output.size(1)

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.spp(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 生成 MobileNet 模型
def generate_mobilenet(num_classes=10, in_channels=1, model_name="MobileNetV2"):
    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # 若后续支持更多 MobileNet 版本，可在此添加
    # 修改输入通道
    model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
    # 修改输出类别数
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name_list = ["CNN", "EfficientCNN", "MobileNetV2", "ShuffleNetV2"]
    for model_name in model_name_list:
        if model_name == "CNN":
            model = CNN(num_classes=10, in_channels=1, input_size=(300, 300))
        elif model_name == "EfficientCNN":
            model = EfficientCNN(num_classes=10, in_channels=1, input_size=(256, 256))
        elif model_name == "MobileNetV2":
            model = generate_mobilenet(num_classes=10, in_channels=1, model_name=model_name)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        param_len = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of model parameters of %s :' % model_name, ' %d ' % param_len)


