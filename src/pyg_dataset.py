import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import pickle
import os
from utils import *
import numpy as np

import sys
# sys.path.insert(1, 'data/')
from tqdm import tqdm

class NetlistDataset(Dataset):
    def __init__(self, data_dir, load_pe = True, split = 1, pl = True, processed=True, graph_indices=None):
        super().__init__()
        self.data_dir = data_dir
        self.data_lst = []

        if graph_indices is None:
            graph_indices = set(int(file_name.split('.')[0]) for file_name in os.listdir(data_dir) if file_name.split('.')[0].isdigit())

        skip_indices = [3, 8, 13, 24, 30, 36, 42, 47, 52, 58, 64, 70] # Skip graphs where the datasets are not complete (missing 'eigen')

        self.load_pe = load_pe

        for graph_index in tqdm(graph_indices):
            if graph_index in skip_indices: continue

            if processed:
                # Get processed data
                data_load_fp = os.path.join('data', 'processed_datasets', f'{str(graph_index)}.pyg_data.pkl') #../
                data = torch.load(data_load_fp)
            else:
                # Get train-test-split
                with open(os.path.join(data_dir, 'split', str(split), f'{str(graph_index)}.split.pkl'), 'rb') as f:
                    split_dict = pickle.load(f)

                train_indices = split_dict['train_indices']
                valid_indices = split_dict['valid_indices']
                test_indices = split_dict['test_indices']

                with open(os.path.join(data_dir, 'split', str(split), f'{str(graph_index)}.split_net.pkl'), 'rb') as f:
                    split_net_dict = pickle.load(f)

                net_train_indices = split_net_dict['train_indices']
                net_valid_indices = split_net_dict['valid_indices']
                net_test_indices = split_net_dict['test_indices']

                # Read node features
                with open(os.path.join(data_dir, f'{str(graph_index)}.node_features.pkl'), 'rb') as f:
                    node_features_dict = pickle.load(f)

                self.design_name = node_features_dict['design']

                num_instances = node_features_dict['num_instances']
                num_nets = node_features_dict['num_nets']
                instance_features = torch.Tensor(node_features_dict['instance_features'])
                instance_features = instance_features[:, [0, 1, 3, 4]] # [(X, Y, cell, cell_width, cell_height, orient)]

                # Get congestion and hpwl targets
                with open(os.path.join(data_dir, f'{str(graph_index)}.targets.pkl'), 'rb') as f:
                    targets_dict = pickle.load(f)

                demand = torch.Tensor(targets_dict['demand'])
                capacity = torch.Tensor(targets_dict['capacity'])
                safe_denominator = torch.where(capacity == 0, torch.ones_like(capacity), capacity)
                result = demand / safe_denominator
                result = torch.where(capacity == 0, torch.ones_like(result), result)
                node_congestion = torch.where(result <= 0.9, torch.zeros_like(result), torch.ones_like(result)).unsqueeze(dim=1).float()
                # node_congestion = (demand-capacity).unsqueeze(dim=1).long()

                with open(os.path.join(data_dir, f'{str(graph_index)}.net_hpwl.pkl'), 'rb') as f:
                    net_hpwl_dict = pickle.load(f)

                net_hpwl = torch.Tensor(net_hpwl_dict['hpwl']).unsqueeze(dim=1).float()

                # Read connection
                with open(os.path.join(data_dir, f'{str(graph_index)}.bipartite.pkl'), 'rb') as f:
                    bipartite_dict = pickle.load(f)

                instance_idx = torch.Tensor(bipartite_dict['instance_idx']).unsqueeze(dim = 1).long()
                net_idx = torch.Tensor(bipartite_dict['net_idx']) + num_instances
                net_idx = net_idx.unsqueeze(dim = 1).long()

                edge_dir = bipartite_dict['edge_dir']
                edge_index_source_to_net = torch.cat(
                    (instance_idx[edge_dir == 0].unsqueeze(0), net_idx[edge_dir == 0].unsqueeze(0)), dim=0
                ).squeeze().long()

                edge_index_sink_to_net = torch.cat(
                    (instance_idx[edge_dir == 1].unsqueeze(0), net_idx[edge_dir == 1].unsqueeze(0)), dim=0
                ).squeeze().long()

                threshold = bipartite_dict['instance_idx'].max() # Remove extraneous edges
                mask = (bipartite_dict['edge_index'][0] <= threshold) & (bipartite_dict['edge_index'][1] <= threshold)
                edge_index = bipartite_dict['edge_index'][:, mask] 
                edge_index_source_sink = torch.Tensor(edge_index).long()

                # Compute degrees
                in_degrees = compute_degrees(edge_index_source_sink, num_instances)
                out_degrees = compute_degrees(torch.flip(edge_index_source_sink, dims=[0]), num_instances)

                source2net_degrees = compute_degrees(edge_index_source_to_net, len(net_hpwl) + num_instances)
                sink2net_degrees = compute_degrees(edge_index_sink_to_net, len(net_hpwl) + num_instances)

                source2net_inst_degrees = source2net_degrees[:num_instances]
                sink2net_inst_degrees = sink2net_degrees[:num_instances]

                source2net_net_degrees = source2net_degrees[num_instances:]
                sink2net_net_degrees = sink2net_degrees[num_instances:]

                # Concatenate node features
                if pl:
                    node_features = np.vstack([in_degrees, out_degrees, source2net_inst_degrees, sink2net_inst_degrees, instance_features[:, 0], instance_features[:, 1], instance_features[:, 2], instance_features[:, 3]]).T  # node_type, instance_features = [node_loc_x, node_loc_y, node_size_x, node_size_y]
                    with open(os.path.join(data_dir, f'{str(graph_index)}.pl_part_dict.pkl'), 'rb') as f:
                        part_dict = pickle.load(f)
                    batch = [part_dict[idx] for idx in range(node_features.shape[0])]
                    num_vn = len(np.unique(batch))
                    batch = torch.tensor(batch).long()

                else:
                    node_features = np.vstack([in_degrees, out_degrees, source2net_inst_degrees, sink2net_inst_degrees, instance_features[:, 2], instance_features[:, 3]]).T # node_type, instance_features[:, 2:] = [node_size_x, node_size_y]
                    batch = None
                    num_vn = 0

                # Read positional encoding
                with open(os.path.join(data_dir, f'{str(graph_index)}.eigen.10.pkl'), 'rb') as f:
                    eig_dict = pickle.load(f)
                    eig_vec = eig_dict['evects'][:num_instances, :]

                if load_pe:
                    node_features = np.concatenate([node_features, eig_vec], axis=1)

                node_features = torch.tensor(node_features).float()
                net_features = torch.tensor(np.vstack([source2net_net_degrees, sink2net_net_degrees]).T).float()

                data = Data(
                    node_features = node_features, 
                    net_features = net_features, 
                    edge_index_source_sink = edge_index_source_sink,
                    edge_index_sink_to_net = edge_index_sink_to_net, 
                    edge_index_source_to_net = edge_index_source_to_net, 
                    node_congestion = node_congestion, 
                    net_hpwl = net_hpwl,
                    batch = batch, 
                    num_vn = num_vn,
                    train_indices = train_indices,
                    valid_indices = valid_indices,
                    test_indices = test_indices,
                    net_train_indices = net_train_indices,
                    net_valid_indices = net_valid_indices,
                    net_test_indices = net_test_indices
                )

                data_save_fp = os.path.join('data', 'processed_datasets', f'{str(graph_index)}.pyg_data.pkl')
                torch.save(data, data_save_fp)
                
            data['design_index'] = graph_index
            self.data_lst.append(data)

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, index):
        x = self.data_lst[index]
        return x
