from collections import OrderedDict

import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet34_Weights, ResNet101_Weights, \
    ResNet152_Weights

"""
我们提供了可能在 FedD3 实验中使用的模型，如下所示：
    - 为 CIFAR-10 定制的 AlexNet 模型（AlexCifarNet），包含 1756426 个参数
    - 为 MNIST 定制的 LeNet 模型，包含 61706 个参数
    - 更多的 ResNet 模型
    - 更多的 VGG 模型
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    supported_dims = {300}    # 导入农业数据集时将该参数改为300

    def __init__(self, num_classes=10, in_channels=1, input_size=(300, 300)):
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
    if model_name == "VGG11":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG11_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG13":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG13_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG16":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG16_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG19":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG19_bn":
        model = models.vgg11_bn(pretrained=True)

    # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    # first_conv_layer.extend(list(model.features))
    # model.features = nn.Sequential(*first_conv_layer)
    # model.conv1 = nn.Conv2d(num_classes, 64, 7, stride=2, padding=3, bias=False)

    fc_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(fc_features, num_classes)

    return model


class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, input_size=(300, 300)):
        super(CNN, self).__init__()

        # 第一组卷积块 - 处理原始尺寸特征
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 150x150
        )

        # 第二组卷积块 - 提取中级特征
        self.group2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 75x75
        )

        # 第三组卷积块 - 提取高级特征
        self.group3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)  # 25x25
        )

        # 第四组卷积块 - 提取深度特征
        self.group4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=5)  # 5x5
        )

        # 空间金字塔池化层 - 提取多尺度特征
        self.spp = nn.AdaptiveAvgPool2d((5, 5))  # 固定输出5x5

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
            nn.Linear(flatten_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
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

class EfficientCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, input_size=(300, 300)):
        super(EfficientCNN, self).__init__()

        # 初始卷积层 - 使用步长2降低尺寸
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # 150x150
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 深度可分离卷积组 - 减少参数量
        self.group1 = nn.Sequential(
            # 深度可分离卷积
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 75x75

            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)  # 25x25
        )

        # 注意力机制模块 - 增强特征选择
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128 // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 // 8, 128, kernel_size=1),
            nn.Sigmoid()
        )

        # 最终特征提取
        self.final = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))  # 固定输出7x7
        )

        # 动态计算全连接层的输入维度
        with torch.no_grad():
            test_input = torch.randn(1, in_channels, *input_size)
            test_output = self.initial(test_input)
            test_output = self.group1(test_output)
            test_output = test_output * self.attention(test_output)  # 应用注意力
            test_output = self.final(test_output)
            test_output = test_output.view(test_output.size(0), -1)
            flatten_size = test_output.size(1)

        # 简化的全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.group1(x)
        x = x * self.attention(x)  # 应用注意力权重
        x = self.final(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
if __name__ == "__main__":
    model_name_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    for model_name in model_name_list:
        model = generate_resnet(num_classes=10, in_channels=1, model_name=model_name)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        param_len = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of model parameters of %s :' % model_name, ' %d ' % param_len)

