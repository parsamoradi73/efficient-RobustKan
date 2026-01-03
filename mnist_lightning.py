from src.efficient_kan.kan import KAN
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from dataclasses import dataclass
from typing import List
import random
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

@dataclass
class Config:
    # Model configuration
    model_type: str 
    hidden_sizes: List[int] 
    activation: str 
    kan_reg_coeff: float 
    kan_smoothness_coeff: float
    kan_activation_coeff: float  # Weight for activation regularization
    kan_entropy_coeff: float  # Weight for entropy regularization
    spline_grid_size: int  # Number of grid points for B-spline basis functions
    
    batch_size: int 
    n_epochs: int 
    learning_rate: float 
    weight_decay: float
    lr_milestones: List[int]
    random_seed: int
    num_workers: int
    
    exp_name: str
    
    def __post_init__(self):
        if self.lr_milestones is None:
            self.lr_milestones = [20]
        
        # Activation function mapping
        self.activation_functions = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU
        }
        
        if self.activation not in self.activation_functions:
            raise ValueError(f"Unknown activation: {self.activation}. Choose from {list(self.activation_functions.keys())}")
        
        self.activation_fn = self.activation_functions[self.activation]

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation after last layer
                layers.append(activation())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class MNISTModule(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        cfg_dict = {k: v for k, v in config.__dict__.items() if k not in ["activation_fn", "activation_functions"]}
        self.save_hyperparameters(cfg_dict)
        self.config = config
        self.model_type = config.model_type
        
        # Model selection
        layer_sizes = [28 * 28] +  config.hidden_sizes + [10]
        if config.model_type.lower() == 'kan':
            self.model = KAN(layer_sizes, grid_size=config.spline_grid_size, random_seed=config.random_seed)
        elif config.model_type.lower() == 'mlp':
            self.model = MLP(layer_sizes, activation=config.activation_fn)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}. Choose 'kan' or 'mlp'")
            
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x.view(-1, 28 * 28))
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config.lr_milestones,
            gamma=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        loss = self.criterion(output, labels)
        if self.model_type == 'kan':
            
            reg_loss = self.config.kan_reg_coeff * self.model.regularization_loss(
                regularize_activation=self.config.kan_activation_coeff,
                regularize_entropy=self.config.kan_entropy_coeff,
                regularize_smoothness=self.config.kan_smoothness_coeff
            )
            total_loss = loss + reg_loss
            # Log regularization loss separately
            self.log('train_reg_loss', reg_loss, prog_bar=True)
        else:
            total_loss = loss
        accuracy = (output.argmax(dim=1) == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        loss = self.criterion(output, labels)
        accuracy = (output.argmax(dim=1) == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        loss = self.criterion(output, labels)
        accuracy = (output.argmax(dim=1) == labels).float().mean()
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', accuracy, prog_bar=True)
        return loss

def main(config: Config):
    # Set random seeds for reproducibility
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.130,), (0.308,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    generator = torch.Generator().manual_seed(config.random_seed)
    train_len = int(0.8 * len(trainset))
    val_len = len(trainset) - train_len
    trainset, valset = torch.utils.data.random_split(trainset, [train_len, val_len], generator=generator)
    
    train_loader = DataLoader(
        trainset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        persistent_workers=True,
    )
    val_loader = DataLoader(
        valset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers, 
        drop_last=False,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        testset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=config.num_workers, 
        persistent_workers=True,
    )

    
    # Model
    model = MNISTModule(config)
    
    if os.path.exists(f'checkpoints/{config.exp_name}'):
        shutil.rmtree(f'checkpoints/{config.exp_name}')
    if os.path.exists(f'lightning_logs/logs/{config.exp_name}'):
        shutil.rmtree(f'lightning_logs/logs/{config.exp_name}')

    # Logger
    logger = TensorBoardLogger("lightning_logs", name=f"logs", version=f"{config.exp_name}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'checkpoints/{config.exp_name}',
            filename=f'mnist-{config.model_type}-{config.activation}-{{epoch:02d}}-{{val_loss:.2f}}',
            save_top_k=1,
            # Removed invalid argument filename_best
            mode='min',
            save_last=True

        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=config.n_epochs,
        accelerator='auto',
        log_every_n_steps=10,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path='last')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan', choices=['kan', 'mlp'],
                      help='Model type to use (kan or mlp)')
    parser.add_argument('--activation', type=str, default='relu',
                      choices=['relu', 'tanh', 'sigmoid', 'leakyrelu', 'elu', 'gelu'],
                      help='Activation function to use (for MLP only)')
    parser.add_argument('--batch-size', type=int, default=1024,
                      help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train')
    parser.add_argument('--lr-milestones', type=int, default=[100], nargs='+',
                      help='List of epochs at which to decay the learning rate')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                      help='Initial learning rate')
    parser.add_argument('--hidden-sizes', type=int, default=[100], nargs='+',
                      help='Size of hidden layers')
    parser.add_argument('--num-workers', type=int, default=os.cpu_count(),
                      help='Number of data loading workers')
    
    parser.add_argument('--weight-decay', type=float, default=0,
                      help='Weight decay (L2 regularization)')
    
    parser.add_argument('--suffix', type=str, default='',
                      help='Suffix for experiment name')

                      
    parser.add_argument('--kan-reg-coeff', type=float, default=1e-3,
                      help='Coefficient for KAN regularization loss')
    parser.add_argument('--kan-smoothness-coeff', type=float, default=0.0,
                      help='Coefficient for KAN smoothness regularization')
    parser.add_argument('--kan-activation-coeff', type=float, default=0.0,
                      help='Weight for activation regularization in KAN')
    parser.add_argument('--kan-entropy-coeff', type=float, default=0.0,
                      help='Weight for entropy regularization in KAN')
    parser.add_argument('--spline-grid-size', type=int, default=5,
                      help='Number of grid points for B-spline basis functions in KAN')

    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    
    args = parser.parse_args()
    
    if args.model == 'kan':
        exp_name = (
            f"{args.model}_G_{args.spline_grid_size}_lr_{args.learning_rate}_hid_{','.join(map(str, args.hidden_sizes))}_wd_{args.weight_decay}"
            f"_reg_{args.kan_reg_coeff}_act_{args.kan_activation_coeff}_ent_{args.kan_entropy_coeff}_smooth_{args.kan_smoothness_coeff}"
            f"_seed_{args.random_seed}{args.suffix}"
        )
    else:
        exp_name = (
            f"{args.model}_hid_{','.join(map(str, args.hidden_sizes))}_lr_{args.learning_rate}_wd_{args.weight_decay}_seed_{args.random_seed}{args.suffix}"
        )

    config = Config(
        model_type=args.model,
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_milestones=args.lr_milestones,
        num_workers=args.num_workers,
        exp_name=exp_name,
        kan_reg_coeff=args.kan_reg_coeff,
        kan_smoothness_coeff=args.kan_smoothness_coeff,
        kan_activation_coeff=args.kan_activation_coeff,
        kan_entropy_coeff=args.kan_entropy_coeff,
        spline_grid_size=args.spline_grid_size,
        random_seed=args.random_seed
    )
    
    main(config) 