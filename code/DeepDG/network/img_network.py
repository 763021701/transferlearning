# coding=utf-8
import torch.nn as nn
from torchvision import models

weights_dict = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet34": models.ResNet34_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "resnet101": models.ResNet101_Weights.DEFAULT,
    "resnet152": models.ResNet152_Weights.DEFAULT,
    "resnext50": models.ResNeXt50_32X4D_Weights.DEFAULT,
    "resnext101": models.ResNeXt101_32X8D_Weights.DEFAULT,
    "vgg11": models.VGG11_Weights.DEFAULT,
    "vgg13": models.VGG13_Weights.DEFAULT,
    "vgg16": models.VGG16_Weights.DEFAULT,
    "vgg19": models.VGG19_Weights.DEFAULT,
    "vgg11bn": models.VGG11_BN_Weights.DEFAULT,
    "vgg13bn": models.VGG13_BN_Weights.DEFAULT,
    "vgg16bn": models.VGG16_BN_Weights.DEFAULT,
    "vgg19bn": models.VGG19_BN_Weights.DEFAULT
}

vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn, "vgg19bn": models.vgg19_bn}


class VGGBase(nn.Module):
    def __init__(self, vgg_name, weights):
        super(VGGBase, self).__init__()
        if weights is None or weights == '':
            model_vgg = vgg_dict[vgg_name](weights=None)
        else:
            model_vgg = vgg_dict[vgg_name](weights=weights_dict[vgg_name])
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d, "resnext101": models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name, weights, freeze_bn):
        super(ResBase, self).__init__()
        if weights is None or weights == '':
            model_resnet = res_dict[res_name](weights=None)
        else:
            model_resnet = res_dict[res_name](weights=weights_dict[res_name])
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

        self.model_resnet = model_resnet
        self.FREEZE_BN = freeze_bn

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.FREEZE_BN:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.model_resnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x
