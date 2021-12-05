from distiller_zoo.KD import DistillKL
from torch import nn

DISTILL_TYPES = {
    'kd': DistillKL(),
    'mse': nn.MSELoss()
}

class DistillationFactory():
    def __init__(self, type):
        self.type = type
        self.distiller = DISTILL_TYPES[type]
