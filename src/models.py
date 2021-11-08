from torchvision import models
from torch import nn
from cifar10_models.resnet import resnet18, resnet34, resnet50

class ModelFactory:
    def __init__(self):
        self.models = {
            'resnet34': Resnet34
        }
    def get_model(self, model_name, **kwargs):
        model = resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, out_features=10)
        return model

class Resnet34:
    def __init__(self, num_classes, pretrained):
        if pretrained:
            return resnet34(pretrained=pretrained)
        else:
            model = resnet34(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
            return model

