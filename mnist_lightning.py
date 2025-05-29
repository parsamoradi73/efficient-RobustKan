from src.efficient_kan.kan import KAN

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from dataclasses import dataclass
from typing import List
import random
import numpy as np

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
    hidden_size: int 
    activation: str 
    kan_reg_coeff: float 
    kan_smoothness_coeff: float
    kan_activation_coeff: float  # Weight for activation regularization
    kan_entropy_coeff: float  # Weight for entropy regularization
    spline_grid_size: int  # Number of grid points for B-spline basis functions
    
    # Training configuration
    batch_size: int 
    n_epochs: int 
    learning_rate: float 
    weight_decay: float
    lr_milestones: List[int]
    random_seed: int
    
    # System configuration
    num_workers: int
    
    # Experiment configuration
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
        self.save_hyperparameters()
        self.config = config
        self.model_type = config.model_type
        
        # Model selection
        layer_sizes = [28 * 28, config.hidden_size, 10]
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
            weight_decay=0 if self.model_type.lower() == 'kan' else self.config.weight_decay
            # weight_decay = self.config.weight_decay
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
            loss += reg_loss
            # Log regularization loss separately
            self.log('train_reg_loss', reg_loss, prog_bar=True)
        accuracy = (output.argmax(dim=1) == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        loss = self.criterion(output, labels)
        accuracy = (output.argmax(dim=1) == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
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
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
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
        persistent_workers=True,
    )
    
    # Model
    model = MNISTModule(config)
    
    # Logger
    logger = TensorBoardLogger("lightning_logs", name=f"logs", version=f"{config.model_type}_{config.exp_name}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'checkpoints/{config.model_type}_{config.exp_name}',
            filename=f'mnist-{config.model_type}-{config.activation}-{{epoch:02d}}-{{val_loss:.2f}}',
            save_top_k=3,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=config.n_epochs,
        accelerator='auto',
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan', choices=['kan', 'mlp'],
                      help='Model type to use (kan or mlp)')
    parser.add_argument('--activation', type=str, default='relu',
                      choices=['relu', 'tanh', 'sigmoid', 'leakyrelu', 'elu', 'gelu'],
                      help='Activation function to use (for MLP only)')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train')
    parser.add_argument('--lr-milestones', type=int, default=[20], nargs='+',
                      help='List of epochs at which to decay the learning rate')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                      help='Initial learning rate')
    parser.add_argument('--hidden-size', type=int, default=100,
                      help='Size of hidden layer')
    parser.add_argument('--num-workers', type=int, default=7,
                      help='Number of data loading workers')

#############################################################################
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay (L2 regularization)')
    
    parser.add_argument('--exp', type=str, default='test',
                      help='Experiment name')

                      
    parser.add_argument('--kan-reg-coeff', type=float, default=1e-3,
                      help='Coefficient for KAN regularization loss')
    

    parser.add_argument('--kan-smoothness-coeff', type=float, default=0.0,
                      help='Coefficient for KAN smoothness regularization')
    parser.add_argument('--kan-activation-coeff', type=float, default=1.0,
                      help='Weight for activation regularization in KAN')
    parser.add_argument('--kan-entropy-coeff', type=float, default=0.0,
                      help='Weight for entropy regularization in KAN')
    

    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    parser.add_argument('--spline-grid-size', type=int, default=5,
                      help='Number of grid points for B-spline basis functions in KAN')
    
    args = parser.parse_args()
    
    config = Config(
        model_type=args.model,
        hidden_size=args.hidden_size,
        activation=args.activation,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_milestones=args.lr_milestones,
        num_workers=args.num_workers,
        exp_name=args.exp,
        kan_reg_coeff=args.kan_reg_coeff,
        kan_smoothness_coeff=args.kan_smoothness_coeff,
        kan_activation_coeff=args.kan_activation_coeff,
        kan_entropy_coeff=args.kan_entropy_coeff,
        spline_grid_size=args.spline_grid_size,
        random_seed=args.random_seed
    )
    
    main(config) 