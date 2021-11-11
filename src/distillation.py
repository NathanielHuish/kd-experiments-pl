from distiller_zoo import KD

distill_types = {
    'KD': KD
}

class Distillation():
    def __init__(self, type):
        return distill_types[type]