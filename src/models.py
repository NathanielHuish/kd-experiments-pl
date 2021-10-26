from torchvision import models
from torch import nn

class ModelFactory:
    def __init__(self):
        self.models = {
            'resnet34': Resnet34
        }
    def get_model(self, model_name, **kwargs):
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, out_features=100)
        return model

class Resnet34:
    def __init__(self, num_classes, pretrained):
        if pretrained:
            return models.resnet34(pretrained=pretrained)
        else:
            model = models.resnet34(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
            return model

