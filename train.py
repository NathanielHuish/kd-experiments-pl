from argparse import ArgumentParser
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from src.training_module import TrainingModule, DistilledTrainingModule
from src.dataset import DataSetFactory
from pl_bolts.datamodules import CIFAR10DataModule
from src.dataset import CIFAR100DataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import json
import os
import wandb

parser = ArgumentParser()
parser = TrainingModule.add_model_specific_args(parser)
parser.add_argument('--train_teacher', type=bool, default=False)
parser.add_argument('--distill', type=bool, default=False)
parser.add_argument('--student_model', type=str, default='resnet34')
parser.add_argument('--teacher_model', type=str, default='resnet34')
parser.add_argument('--prune_target', type=float, default=0.0)
parser.add_argument('--model_name', type=str)
parser.add_argument('--precision', type=int)



# trainer arguments
parser.add_argument('--default_root_dir', type=str, default='logs')
parser.add_argument('--epochs', type=int, default=240)
parser.add_argument('--gpus', type=int, default=(1 if th.cuda.is_available() else 0))
parser.add_argument('--batch_size', type=int, default=2084)
parser.add_argument('--num_workers', type=int, default=4)


def main(args):
    sweep_config = {
        "name" : "first-distill",
        "method" : "random",

        "parameters" : {
            "epochs" : {
            "values" : [120]
            },
        "learning_rate" :{
            "min": 0.0001,
            "max": 0.05
        },
        "weight_decay":{
            "min": 1e-5,
            "max": 1e-3
        },
        "precision":{
            "values" : [16, 32]
        },
        "mixup": {
            "values" : [True, False]
        }
    }
    }

    wandb.init(config=args)
    sweep_id = wandb.sweep(sweep_config, project="cifar100-sweep")
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    
    if args.train_teacher:
        training_module = TrainingModule(
            model_name='resnet34',
            image_size=args.image_size,
            num_classes=args.num_classes,
            pre_trained=args.pre_trained,
            lr=config['lr'],
            epochs=config['epochs'],
            mixup=config['mixup'],
            momentum=args.momentum,
            weight_decay=config['weight_decay']
        )
    elif args.distill:
        training_module = DistilledTrainingModule(
            student_model_name='resnet34',
            teacher_model_name='resnet34',
            image_size=args.image_size,
            num_classes=args.num_classes,
            lr=config['lr'],
            momentum=args.momentum,
            epochs=config['epochs'],
            weight_decay=config['weight_decay'],
            mixup=config['mixup'],
            pretrained=args.pre_trained
        )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    wandb_logger = WandbLogger()
    trainer = pl.Trainer.from_argparse_args(
        args, 
        max_epochs=config['epochs'],
        precision=16, 
        logger=wandb_logger,
        callbacks=[lr_monitor])
        
    dm = CIFAR100DataModule(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    trainer.fit(training_module, datamodule=dm)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
