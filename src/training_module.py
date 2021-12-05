from argparse import ArgumentParser
import pytorch_lightning as pl
import torch as th
from torch import nn
import torchmetrics
import torch_optimizer as optim
import numpy as np
from src.distillation import DistillationFactory
import timm


from src.models import ModelFactory
from src.dataset import DataSetFactory

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = th.randperm(batch_size).cuda()
    else:
        index = th.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class TrainingModule(pl.LightningModule):
    def __init__(
        self, 
        model_name, 
        image_size, 
        num_classes, 
        lr, 
        momentum, 
        epochs,
        weight_decay,
        mixup,
        pre_trained=False,
    ):
        super(TrainingModule, self).__init__()
        self.lr = lr
        self.image_size = image_size
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.epochs = epochs
        self.mixup = mixup
        self._model = self.create_model(model_name=model_name, pre_trained=pre_trained)
        self._loss = nn.CrossEntropyLoss()
        acc = torchmetrics.Accuracy()
        self.val_acc = acc.clone()
        self.train_acc = acc.clone()
        self.save_hyperparameters()

    def create_model(self, model_name, pre_trained):
        model_factory = ModelFactory()
        model = model_factory.get_model(model_name=model_name,
                                        image_size=self.image_size,
                                        num_classes=self.num_classes,
                                        pre_trained=pre_trained)
        return model

    def forward(self, images):
        return self._model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = None
        if self.mixup:
            images, targets_a, targets_b, lam = mixup_data(images, labels,
                                                        1, use_cuda=True)
            y_hat = self.forward(images)
            loss = mixup_criterion(self._loss, y_hat, targets_a, targets_b, lam)
            loss = loss.mean() 
        else:
            y_hat = self.forward(images)
            loss = self._loss(y_hat, labels).mean()
        self.train_acc(y_hat, labels)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss.detach())
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self.forward(images)
        loss = self._loss(y_hat, labels)
        self.val_acc(y_hat, labels)
        self.log('valid_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('valid_loss', loss.detach(), on_step=True, on_epoch=True)
        return {'loss': loss}


    def configure_optimizers(self):
        optimizer =  th.optim.Adam(self._model.parameters(),
                         lr=self.lr, weight_decay=self.weight_decay)

        # lr_scheduler = th.optim.lr_scheduler.OneCycleLR(
        #                                         optimizer, 
        #                                         max_lr=self.lr,
        #                                         total_steps=self.num_training_steps
        #                                     )
        lr_scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.epochs//2, (self.epochs-(self.epochs//4))])

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "valid_loss"}
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-1)
        parser.add_argument('--momentum', type=float, default=0.0)
        parser.add_argument('--image_size', type=int, default=32)
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument('--pre_trained', type=bool, default=False)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--mixup', type=bool, default=True)
        return parser

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

class DistilledTrainingModule(TrainingModule):
    def __init__(self, 
                model_name, 
                teacher_model_name, 
                image_size,
                num_classes,
                lr,
                momentum, #add alpha, beta (or kd loss, label loss)
                epochs,
                weight_decay,
                mixup,
                soft_loss,
                hard_loss,
                distill,
                pretrained=False,
    ):
        super(DistilledTrainingModule, self).__init__(model_name=model_name, 
                                                      image_size=image_size, 
                                                      num_classes=num_classes, 
                                                      lr=lr, 
                                                      momentum=momentum, 
                                                      epochs=epochs,
                                                      weight_decay=weight_decay,
                                                      mixup=mixup,
                                                      pre_trained=pretrained)
        self.model_name = model_name
        self.teacher_model_name = teacher_model_name
        self.soft_loss = soft_loss
        self.hard_loss = hard_loss
        self.lr = lr
        self.image_size = image_size
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.epochs = epochs
        self.mixup = mixup
        self._mse_loss = nn.MSELoss()
        self.distill = distill
        self.critereon = DistillationFactory(self.distill)
        self._teacher_model = self.create_model(self.teacher_model_name, pre_trained=False)
        #self._teacher_model.load()
        for param in self._teacher_model.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        images, labels = batch
        y_hat_student = self.forward(images)
        y_hat_teacher = self._teacher_model.forward(images)

        kd_loss = self.critereon.distiller(y_hat_student, y_hat_teacher)
        self.log('kd_loss', kd_loss)

        normal_loss = nn.CrossEntropyLoss()(y_hat_student, labels)
        self.log('normal_loss', normal_loss)

        loss = self.soft_loss * kd_loss + self.hard_loss * normal_loss
        return {'loss': loss} #alpha * kd_loss + beta * normal_loss

