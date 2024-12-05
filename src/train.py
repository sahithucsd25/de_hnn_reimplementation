import torch
import torch.nn
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import *
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv

from pyg_dataset import NetlistDataset
from model import GNN_node

def train(test=False, n_epochs=100):
    dataset = NetlistDataset(data_dir='../data/processed_datasets', graph_indices=[40])

    # Organize features in dataset
    h_dataset = []
    for data in tqdm(dataset):
        num_instances = data.node_congestion.shape[0]
        data.num_instances = num_instances
        data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances
        data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances

        h_data = HeteroData()
        h_data['node'].x = data.node_features
        h_data['node'].y = data.node_congestion
        
        h_data['net'].x = data.net_features
        h_data['net'].y = data.net_hpwl
        
        h_data['node', 'as_a_sink_of', 'net'].edge_index, h_data['node', 'as_a_sink_of', 'net'].edge_weight = gcn_norm(data.edge_index_sink_to_net, add_self_loops=False)
        h_data['node', 'as_a_source_of', 'net'].edge_index = data.edge_index_source_to_net

        h_data.batch = data.batch
        h_data.num_vn = data.num_vn
        h_data.num_instances = num_instances

        h_data.train_indices = data.train_indices
        h_data.valid_indices = data.valid_indices
        h_data.test_indices = data.test_indices
        h_data.net_train_indices = data.net_train_indices
        h_data.net_valid_indices = data.net_valid_indices
        h_data.net_test_indices = data.net_test_indices

        h_dataset.append(h_data)

    device = "cpu"

    model = GNN_node(4, 32, 8, 1, node_dim = data.node_features.shape[1], net_dim = data.net_features.shape[1], vn = True).to(device) #, vn=True

    criterion_node = nn.BCEWithLogitsLoss()
    criterion_net = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_total_val = None

    losses_all = {}

    if test:
        # Load existing model
        model = torch.load("results/best_dehnn_model.pt")
    else:
        # Training loop
        for epoch in range(n_epochs):
            loss_node_all = 0
            loss_net_all = 0
            val_loss_node_all = 0
            val_loss_net_all = 0

            losses_all[str(epoch)] = {} # Dictionary to track loss for node/net per split
            train_flawless = 0
            model.train()
            # Training loop
            for chip in tqdm(h_dataset):
                try:
                    optimizer.zero_grad()
                    node_representation, net_representation = model(chip, device)
                    node_representation, net_representation = node_representation[chip.train_indices, :], net_representation[chip.net_train_indices, :].squeeze(dim=0)
                    loss_node = criterion_node(node_representation, chip['node'].y[chip.train_indices, :].to(device))
                    loss_net = criterion_net(net_representation, chip['net'].y[chip.net_train_indices, :].to(device))
                    loss = loss_node + 0.001*loss_net
                    loss.backward()
                    optimizer.step()
                    train_flawless += 1
                except:
                    print('OOM')
                    continue

                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()

            losses_all[str(epoch)]['train'] = [loss_node_all/train_flawless, loss_net_all/train_flawless]

            valid_flawless = 0
            model.eval()
            # Validation Loop
            for chip in tqdm(h_dataset):
                try:
                    node_representation, net_representation = model(chip, device)
                    node_representation, net_representation = node_representation[chip.valid_indices, :], net_representation[chip.net_valid_indices, :].squeeze(dim=0)
                    val_loss_node = criterion_node(node_representation, chip['node'].y[chip.valid_indices, :].to(device))
                    val_loss_net = criterion_net(net_representation, chip['net'].y[chip.net_valid_indices, :].to(device))

                    val_loss_node_all += val_loss_node.item()
                    val_loss_net_all += val_loss_net.item()
                    valid_flawless += 1
                except:
                    print("OOM")
                    continue
            
            if (best_total_val is None) or ((val_loss_node_all/valid_flawless) < best_total_val):
                # Save the model if it has the lowest loss so far
                best_total_val = val_loss_node_all/valid_flawless
                torch.save(model, 'results/best_dehnn_model.pt')

            losses_all[str(epoch)]['valid'] = [val_loss_node_all/valid_flawless, val_loss_net_all/valid_flawless]

            if (epoch+1)%10==0: print(f'Epoch {epoch+1} completed')

    # Compute evaluation metrics
    train_f1, train_net_l1 = evaluate_model(model, h_dataset, 'train', device)
    valid_f1, valid_net_l1 = evaluate_model(model, h_dataset, 'valid', device)
    test_f1, test_net_l1 = evaluate_model(model, h_dataset, 'test', device)

    if not test:
        # Write losses per epoch to results
        with open('results/losses_node.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Train Loss', 'Valid Loss'])
            for epoch_idx, losses in losses_all.items():
                writer.writerow([epoch_idx, losses['train'][0], losses['valid'][0]])

        with open('results/losses_net.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Train Loss', 'Valid Loss'])
            for epoch_idx, losses in losses_all.items():
                writer.writerow([epoch_idx, losses['train'][1], losses['valid'][1]])
        
        print('Losses saved.')

    # Write evaluation metrics to results
    with open('results/eval_metrics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Split', 'F1 Score', 'L1 Loss'])
        writer.writerow(['Train', train_f1, train_net_l1])
        writer.writerow(['Valid', valid_f1, valid_net_l1])
        writer.writerow(['Test', test_f1, test_net_l1])
        print('Evaluation metrics saved.')