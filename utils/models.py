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

    def __init__(self, num_classes=10, in_channels=1, input_size=(300, 300)):   # 导入农业数据集时将input_size改为(300, 300)
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

        self.fp_con1 = nn.Sequential(OrderedDict([
            ('con0', nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        self.ternary_con2 = nn.Sequential(OrderedDict([
            # 卷积层模块 1
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),

            # 卷积层模块 2
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),

            # 卷积层模块 3
            ('conv3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        # 提前计算全连接层的输入维度
        with torch.no_grad():
            test_input = torch.randn(1, in_channels, *input_size)
            test_output = self.fp_con1(test_input)
            test_output = self.ternary_con2(test_output)
            test_output = test_output.view(test_output.size(0), -1)
            flatten_size = test_output.size(1)

        self.fp_fc = nn.Linear(flatten_size, num_classes, bias=False)

    def forward(self, x):
        x = self.fp_con1(x)
        x = self.ternary_con2(x)
        x = x.view(x.size(0), -1)
        x = self.fp_fc(x)
        output = F.log_softmax(x, dim=1)
        return output

if __name__ == "__main__":
    model_name_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    for model_name in model_name_list:
        model = generate_resnet(num_classes=10, in_channels=1, model_name=model_name)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        param_len = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of model parameters of %s :' % model_name, ' %d ' % param_len)

