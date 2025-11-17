from collections import OrderedDict

import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, \
    ResNet18_Weights

"""
我们提供了可能在 FedD3 实验中使用的模型，如下所示：
    - 为 CIFAR-10 定制的 AlexNet 模型（AlexCifarNet），包含 1756426 个参数
    - 为 MNIST 定制的 LeNet 模型，包含 61706 个参数
    - 更多的 ResNet 模型
    - 更多的 VGG 模型
"""


# 为 CIFAR-10 定制的 AlexNet 模型，包含 1756426 个参数
class AlexCifarNet(nn.Module):
    supported_dims = {32}

    def __init__(self):
        super(AlexCifarNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), 4096)
        out = self.classifier(out)
        return out


# 为 MNIST 定制的 LeNet 模型，包含 61706 个参数
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    supported_dims = {28}  # 导入农业数据集时将该参数改为300

    def __init__(self, num_classes=10, in_channels=1, input_size=(28, 28)):  # 导入农业数据集时将input_size改为(300, 300)
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
    else:
        raise ValueError(f"不支持的 ResNet 模型: {model_name}")
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    return model


class BasicBlock(nn.Module):
    expansion = 1  # 残差块输出通道数是输入的1倍（区别于Bottleneck的4倍）

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Args:
            inplanes: 输入通道数
            planes: 中间卷积通道数（最终输出通道数=planes×expansion）
            stride: 卷积步长（用于降维）
            downsample:  shortcut分支（用于通道/尺寸不匹配时调整维度）
        """
        super(BasicBlock, self).__init__()
        # 第一个3×3卷积（带BN和ReLU）
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)  # padding=1保持尺寸不变（3×3卷积）
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # 第二个3×3卷积（仅带BN，ReLU在残差相加后）
        self.conv2 = nn.Conv2d(planes, planes * self.expansion, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # 残差分支（原始输入）

        # 主分支：卷积→BN→ReLU→卷积→BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # shortcut分支：若通道/尺寸不匹配，用1×1卷积调整
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差相加 + 激活
        out += residual
        out = self.relu(out)

        return out


# 针对CIFAR-10优化的ResNet-18
class ResNet18_n(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_n, self).__init__()
        self.inplanes = 64  # 初始通道数（比原始ResNet-18更早进入64通道，增强特征提取）

        # 主干网络起始层（适配32×32输入）
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1,
                               padding=1, bias=False)  # 3×3卷积（无步长，无池化）
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 四个残差阶段（每个阶段的block数：ResNet-18固定为[2,2,2,2]）
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)  # 32×32 → 32×32（通道64）
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # 32×32 → 16×16（通道128）
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # 16×16 → 8×8（通道256）
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  # 8×8 → 4×4（通道512）

        # 全局平均池化（适配任意尺寸，输出1×1特征图）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout（缓解CIFAR-10过拟合）
        self.dropout = nn.Dropout(p=0.5)
        # 全连接层（512维特征→10类）
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # 权重初始化（提升训练稳定性）
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建残差阶段（由多个BasicBlock组成）
        Args:
            block: 残差块类型（BasicBlock）
            planes: 中间通道数
            blocks: 该阶段的block数量
            stride: 第一个block的步长（用于降维）
        """
        downsample = None
        # 当步长≠1 或 输入通道≠输出通道时，需要shortcut调整维度
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),  # 1×1卷积降维/升维
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个block（可能带downsample和stride）
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # 更新输入通道数（后续block复用）
        # 剩余block（步长=1，无downsample）
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """权重初始化：卷积层用Kaiming正态分布，BN层初始化gamma=1、beta=0"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 前向传播流程：输入→卷积→BN→ReLU→残差阶段×4→全局池化→Dropout→全连接
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # 4×4→1×1（输出shape: [B, 512, 1, 1]）
        x = torch.flatten(x, 1)  # 扁平化→[B, 512]
        x = self.dropout(x)  # 随机失活（缓解过拟合）
        x = self.fc(x)  # 输出[B, 10]（10类分数）

        return x


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
    else:
        raise ValueError(f"不支持的 VGG 模型: {model_name}")

    first_conv_layer = [nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(list(model.features))
    model.features = nn.Sequential(*first_conv_layer)
    model.conv1 = nn.Conv2d(num_classes, 64, 7, stride=2, padding=3, bias=False)

    fc_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(fc_features, num_classes)

    return model


class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
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
            # nn.Dropout2d(p=0.05),

            # 卷积层模块 3
            ('conv3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.fp_fc = nn.Linear(4096, num_classes, bias=False)

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
