from utils.models import *


def assign_dataset(dataset_name):
    """
    为数据集分配参数
    :param dataset_name: 数据集名称
    :return: num_class: 数据集中的类别数量
    :return: image_dim: 图像尺寸
    :return: image_channel: 图像通道数
    """
    num_class = -1
    image_dim = -1
    image_channel = -1

    if dataset_name == 'MNIST':
        num_class = 10
        image_dim = 28
        image_channel = 1

    elif dataset_name == 'FashionMNIST':
        num_class = 10
        image_dim = 28
        image_channel = 1

    elif dataset_name == 'EMNIST':
        num_class = 27
        image_dim = 28
        image_channel = 1

    elif dataset_name == 'CIFAR10':

        num_class = 10
        image_dim = 32
        image_channel = 3

    elif dataset_name == 'CIFAR100':

        num_class = 100
        image_dim = 32
        image_channel = 3

    elif dataset_name == 'SVHN':

        num_class = 10
        image_dim = 32
        image_channel = 3

    elif dataset_name == 'IMAGENET':

        num_class = 200
        image_dim = 64
        image_channel = 3

    return num_class, image_dim, image_channel


def init_model(model_name, num_class, image_channel):
    """
    为特定的学习任务初始化模型。
    :param model_name: 模型名称
    :param num_class: 数据集中的类别数量
    :param image_channel: 图像通道数
    :return: 初始化后的模型
    """
    model = None
    if model_name == "ResNet18":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet50":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet34":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet101":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet152":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "LeNet":
        model = LeNet(num_classes=num_class, in_channels=image_channel)
    elif model_name == "CNN":
        model = CNN(num_classes=num_class, in_channels=image_channel)
    elif model_name == "VGG11":
        model = generate_vgg(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "VGG11_bn":
        model = generate_vgg(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "AlexCifarNet":
        model = AlexCifarNet()
    else:
        print('Model is not supported')

    return model