from torchvision import models
from torch import nn
import torch
import pytorch_lightning as pl
from cifar10_models.resnet import resnet18, resnet34, resnet50

class ModelFactory():
    def __init__(self):
        self.models = {
        }

    def get_model(self, model_name, **kwargs):
        if model_name == 'resnet18':
            model = resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, out_features=10)
            return model
        elif model_name == 'resnet34':
            model = resnet34(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, out_features=10)
            return model
        elif model_name not in self.models:
            model = resnet34(pretrained=False)
            checkpoint = torch.load(model_name)

            dictionary = checkpoint['state_dict']
            for key in list(dictionary.keys()):
                old = key
                new = key.replace('_model.', '')
                dictionary[new] = dictionary.pop(old)

            model.load_state_dict(dictionary)
            return model
        else:
            model = self.models[model_name](num_classes=10, pretrained=False)
            return model

