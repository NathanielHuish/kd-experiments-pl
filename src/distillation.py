from distiller_zoo import KD
from torch import nn

DISTILL_TYPES = {
    'kd': KD,
    'mse': nn.MSELoss()
}

class DistillationFactory():
    def __init__(self, type):
        self.type = type
        self.distiller = DISTILL_TYPES[type].cuda()
