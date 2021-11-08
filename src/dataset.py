from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
import pytorch_lightning as pl
from pathlib import Path

DATASET_PATH = Path('./data')

class DataSetFactory:
    def __init__(self, hparams):
        self.hparams = hparams
        self.dataset = None
        self.datasets = {
            'cifar10': CIFAR10DataModule,
            'cifar100': CIFAR100DataModule,
            'imagenet': ImagenetDataModule
        }
        check_data_dir()
        
        self.dataset = self.datasets[self.hparams.dataset](batch_size=self.hparams.batch_size)

        if self.dataset is None:
            raise NotImplementedError('Dataset not implemented')
      
    def get_train_dataloader(self):
        return self.dataset.train_dataloader()

    def get_valid_dataloader(self):
        return self.dataset.val_dataloader()


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, pin_memory):
        super().__init__()
        self.data_dir = DATASET_PATH
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    
    @property
    def num_classes(self):
        return 100

    #def prepare_data():
        #datasets.CIFAR100.download()

    def setup(self, stage):
        #process and split
        return 0
    
    def train_dataloader(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        dataset = datasets.CIFAR100(root=self.data_dir, 
                                    download=True,
                                    train=True,
                                    transform=train_transform)
        
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return loader


    def val_dataloader(self):
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        dataset = datasets.CIFAR100(root=self.data_dir, 
                                    download=True, 
                                    train=False, 
                                    transform=test_transform)
        
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)
        
        return loader

    def test_dataloader(self):
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        dataset = datasets.CIFAR100(root=self.data_dir, 
                                    download=True, 
                                    train=False, 
                                    transform=test_transform)
        
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False)
        
        return loader

def check_data_dir():
    if not DATASET_PATH.exists() and not DATASET_PATH.is_dir():
        Path.mkdir(DATASET_PATH)