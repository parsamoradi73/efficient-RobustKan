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

def fgsm_attack(model, x, y, epsilon, device):
    """
    Perform FGSM attack on the model.
    
    Args:
        model: The model to attack
        x: Input tensor
        y: Target tensor
        epsilon: Attack strength
        device: Device to run the attack on
    
    Returns:
        Perturbed input tensor
    """
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    
    x.requires_grad = True
    
    # Forward pass
    output = model(x)
    loss = F.cross_entropy(output, y)
    
    # Backward pass
    loss.backward()
    
    # Create perturbation
    perturbation = epsilon * torch.sign(x.grad.data)
    
    # Create adversarial example
    x_adv = x + perturbation
    # Clamp to valid image range [-1, 1] (assuming normalized data)
    x_adv = torch.clamp(x_adv, -1, 1)
    
    return x_adv

def evaluate_robustness(model, test_loader, epsilons, device):
    """
    Evaluate model robustness against FGSM attack with different epsilon values.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        epsilons: List of epsilon values to test
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing accuracies for each epsilon
    """
    model.eval()
    results = {eps: [] for eps in epsilons}
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        
        # Test clean accuracy
        with torch.no_grad():
            output = model(x)
            pred = output.argmax(dim=1)
            clean_correct = (pred == y).float().mean().item()
            results[0.0].append(clean_correct)
        
        # Test adversarial accuracy
        for eps in epsilons:
            if eps == 0.0:
                continue
                
            x_adv = fgsm_attack(model, x, y, eps, device)
            
            with torch.no_grad():
                output = model(x_adv)
                pred = output.argmax(dim=1)
                adv_correct = (pred == y).float().mean().item()
                results[eps].append(adv_correct)
    
    # Calculate mean accuracies
    mean_results = {eps: np.mean(accs) for eps, accs in results.items()}
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
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=7,
                      help='Number of data loading workers')
    parser.add_argument('--epsilons', type=float, nargs='+',
                      default=[0.0, 2.0/255, 4.0/255, 8.0/255, 16.0/255, 64.0/255],
                      help='Epsilon values for FGSM attack')
    parser.add_argument('--exp', type=str, default='test',
                      help='Experiment name')
    args = parser.parse_args()

    # Load the checkpoint
    # checkpoint = torch.load(args.checkpoint)
    base_dir = f"lightning_logs/logs/{args.exp}"
    print(os.path.join(base_dir, 'hparams.yaml'))
    with open(os.path.join(base_dir, 'hparams.yaml'), 'r') as file:
        cfg = yaml.safe_load(file)['config']
    
    config = Config(
        model_type=cfg['model_type'],
        hidden_size=cfg['hidden_size'],
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
    model = MNISTModule.load_from_checkpoint(args.checkpoint, config=config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Evaluate robustness
    print("Evaluating model robustness...")
    results = evaluate_robustness(model, test_loader, args.epsilons, device)
    
    # Print results
    print("\nResults:")
    print("-" * 40)
    print("Epsilon  |  Accuracy")
    print("-" * 40)
    for eps, acc in results.items():
        print(f"{eps:.3f}    |  {acc:.4f}")
    
    # # Plot results
    # plot_results(results, f"Model Robustness Against FGSM Attack ({config.model_type})")
    # print("\nPlot saved as 'robustness_plot.png'")

    # Save numerical results
    # np.save('robustness_results.npy', results)
    # print("Numerical results saved as 'robustness_results.npy'")

if __name__ == "__main__":
    main() 