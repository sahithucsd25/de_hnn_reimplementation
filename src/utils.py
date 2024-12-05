import torch
import torch.nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def compute_binary_accuracy(logits, targets):
    # Convert logits to binary predictions (0 or 1)
    predictions = (logits >= 0.9).long()
    
    # Compare predictions to targets and compute accuracy
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    
    return correct / total

def compute_binary_f1_score(logits, targets):
    # Convert logits to binary predictions (0 or 1)
    predictions = (logits >= 0.9).long()
    
    # True Positives (TP)
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    
    # False Positives (FP)
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    
    # False Negatives (FN)
    fn = ((predictions == 0) & (targets == 1)).sum().item()
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1

def compute_degrees(edge_index, num_nodes=None):
    # If num_nodes is not provided, infer it from edge_index
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    # Create a degree tensor initialized to zero
    degree = torch.zeros(num_nodes, dtype=torch.long)
    
    # Count the number of edges connected to each node
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
    
    return degree

def evaluate_model(model, dataset, split, device):
    # Compute F1 and MAE
    total_f1 = 0
    total_net_l1 = 0

    model.eval()
    for chip in tqdm(dataset):
        indices_dict = {
            'train': {
                'inst': chip.train_indices,
                'net': chip.net_train_indices
            },
            'valid': {
                'inst': chip.valid_indices,
                'net': chip.net_valid_indices
            },
            'test': {
                'inst': chip.test_indices,
                'net': chip.net_test_indices
            }
        }
        indices = indices_dict[split]['inst']
        net_indices = indices_dict[split]['net']
        try:
            node_representation, net_representation = model(chip, device)
            node_representation, net_representation = node_representation[indices, :], net_representation[net_indices, :].squeeze(dim=0)
            total_f1 += compute_binary_f1_score(node_representation, chip['node'].y[indices, :].to(device))
            total_net_l1 += torch.nn.functional.l1_loss(net_representation, chip['net'].y[net_indices, :].to(device))
        except:
            print('OOM')
            continue

    return total_f1/len(dataset), total_net_l1.item()/len(dataset)

def plot_losses(comp):
    data = pd.read_csv(f'results/losses_{comp}.csv')
    capitalized = comp[0].upper() + comp[1:] 
    loss = 'MSE' if comp=='net' else 'Binary Cross-Entropy'

    plt.rcParams.update({
        'font.size': 14,  # General font size
        'axes.titlesize': 18,  # Title font size
        'axes.labelsize': 16,  # X and Y label size
        'xtick.labelsize': 14,  # X-axis tick label size
        'ytick.labelsize': 14,  # Y-axis tick label size
        'legend.fontsize': 14  # Legend font size
    })

    # Plotting the losses
    plt.figure(figsize=(10, 6))
    plt.plot(data['Epoch'], data['Train Loss'], label='Train Loss') # , marker='o'
    plt.plot(data['Epoch'], data['Valid Loss'], label='Validation Loss') # , marker='o'
    plt.xlabel('Epoch')
    plt.ylabel(f'{loss} Loss')
    plt.title(f'{capitalized} Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Save the plot in the 'visualizations' folder
    plt.savefig(f'plots/{comp}_loss.png')
