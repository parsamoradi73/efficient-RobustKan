import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import lightning as L
from mnist_lightning import MNISTModule, Config
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml


def pgd_attack(model, x, y, device, epsilon, steps=10):
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    x_adv = x.clone().detach()

    step_size = epsilon / 5.0

    for _ in range(steps):
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        if x_adv.grad is not None: 
            x_adv.grad.zero_()
        loss.backward()
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + step_size * grad_sign
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        # x_adv = torch.clamp(x_adv, -1, 1)
        x_adv = x_adv.detach()

    return x_adv

def fgsm_attack(model, x, y, epsilon, device):
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    
    x.requires_grad = True
    
    output = model(x)
    loss = F.cross_entropy(output, y)
    
    loss.backward()
    
    perturbation = epsilon * torch.sign(x.grad.data)
    x_adv = x + perturbation
    # x_adv = torch.clamp(x_adv, -1, 1)
    x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
    
    return x_adv

def evaluate_robustness(model, test_loader, epsilons, device, attack, steps, step_size):
    model.eval()
    results = {eps: 0 for eps in epsilons}
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        
        if attack.lower() == 'fgsm':
            for eps in epsilons:
                x_adv = fgsm_attack(model, x, y, eps, device)
                with torch.no_grad():
                    output = model(x_adv)
                    pred = output.argmax(dim=1)
                    adv_correct = (pred == y).float().sum().item()
                    results[eps] += adv_correct
    
        elif attack.lower() == 'pgd':
            for eps in epsilons:
                x_adv = pgd_attack(model, x, y, device, eps, steps)
                with torch.no_grad():
                    output = model(x_adv)
                    pred = output.argmax(dim=1)
                    adv_correct = (pred == y).float().sum().item()
                    results[eps] += adv_correct
    mean_results = {eps: accs / len(test_loader.dataset) for eps, accs in results.items()}

    return mean_results

def plot_results(results, title="Model Robustness Against FGSM Attack"):
    """
    Plot the results of robustness evaluation.
    
    Args:
        results: Dictionary containing accuracies for each epsilon
        title: Plot title
    """
    epsilons = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies, marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.savefig('robustness_plot.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='last',
                      help='Batch size for testing')
    parser.add_argument('--batch-size', type=int, default=1024,
                      help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=os.cpu_count(),
                      help='Number of data loading workers')
    parser.add_argument('--epsilons', type=float, nargs='+',
                      default=[0, 2.0/255, 4.0/255, 8.0/255, 16.0/255, 32.0/255, 64.0/255, 96.0/255],
                      help='Epsilon values for FGSM attack')
    parser.add_argument('--exp', type=str, default='test',
                      help='Experiment name')
    parser.add_argument('--steps', type=int, default=40,
                      help='Number of steps for PGD attack')
    parser.add_argument('--step-size', type=float, default=0.01,
                      help='Step size for PGD attack')
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'pgd'],
                      help='Attack type')
    args = parser.parse_args()

    # Load the checkpoint
    # checkpoint = torch.load(args.checkpoint)
    base_dir = f"lightning_logs/logs/{args.exp}"
    print(os.path.join(base_dir, 'hparams.yaml'))
    with open(os.path.join(base_dir, 'hparams.yaml'), 'r') as file:
        cfg = yaml.safe_load(file)
    
    config = Config(
        model_type=cfg['model_type'],
        hidden_sizes=cfg['hidden_sizes'],
        activation=cfg['activation'],
        batch_size=cfg['batch_size'],
        n_epochs=cfg['n_epochs'],
        learning_rate=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
        lr_milestones=cfg['lr_milestones'],
        num_workers=cfg['num_workers'],
        exp_name=cfg['exp_name'],
        kan_reg_coeff=cfg['kan_reg_coeff'],
        kan_smoothness_coeff=cfg['kan_smoothness_coeff'],
        kan_activation_coeff=cfg['kan_activation_coeff'],
        kan_entropy_coeff=cfg['kan_entropy_coeff'],
        spline_grid_size=cfg['spline_grid_size'],
        random_seed=cfg['random_seed']
    )

    # Create model and load state
    # model = MNISTModule(config)
    if args.checkpoint == 'last':
        model = MNISTModule.load_from_checkpoint(f"checkpoints/{args.exp}/last.ckpt", config=config)
    else:
        model = MNISTModule.load_from_checkpoint(args.checkpoint, config=config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.130,), (0.308,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=False,
    )

    # Evaluate robustness
    print("Evaluating model robustness...")
    results = evaluate_robustness(model, test_loader, args.epsilons, device, args.attack, args.steps, args.step_size)
    # Print results
    print("\nResults:")
    print("-" * 40)
    print("Epsilon  |  Accuracy")
    print("-" * 40)
    for eps, acc in results.items():
        print(f"{eps:.3f}    |  {acc:.4f}")

if __name__ == "__main__":
    main() 