import torchvision.models as models

vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn, "vgg19bn": models.vgg19_bn}

res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d, "resnext101": models.resnext101_32x8d}

available_weights = models.ResNet18_Weights
print("Available ResNet18 weights:")
for weight in available_weights:
    print(weight)

available_weights = models.ResNet50_Weights
print("Available ResNet50 weights:")
for weight in available_weights:
    print(weight)

available_weights = models.VGG16_Weights
print("Available VGG16 Weights:")
for weight in available_weights:
    print(weight)



#resnet50 = models.resnet50(pretrained=False)
#print(resnet50)
#resnet18 = models.resnet18(pretrained=False)
#print(resnet18)

vgg16 = models.vgg16(pretrained=False)
print(vgg16)

"""
for net_name in vgg_dict.keys():
    net = vgg_dict[net_name](pretrained=False)
    num_params = sum(param.numel() for param in net.parameters())
    print(net_name, num_params)

for net_name in res_dict.keys():
    net = res_dict[net_name](pretrained=False)
    num_params = sum(param.numel() for param in net.parameters())
    print(net_name, num_params)
"""
