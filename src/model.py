import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Sequential as Seq, Linear, ReLU

from dehnn_layers import HyperConvLayer

from torch_geometric.utils.dropout import dropout_edge

class GNN_node(nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, out_node_dim, out_net_dim,
                        node_dim = None, 
                        net_dim = None, 
                        vn = False, 
                        device = 'cuda'
                    ):
        '''
            num_layer (int): number of GNN message passing layers
            emb_dim (int): node embedding dimensionality
            out_node_dim (int): node output dimensionality (which is then passed through linear layer for single output)
            out_net_dim (int): net output dimensionality
            node_dim (int): node input dimensionality
            net_dim (int): net input dimensionality
            vn (bool): whether or not virtual nodes should be passed through the model
        '''

        super(GNN_node, self).__init__()
        self.device = device

        self.num_layer = num_layer

        self.node_dim = node_dim
        self.net_dim = net_dim
        self.emb_dim = emb_dim
        self.out_node_dim = out_node_dim
        self.out_net_dim = out_net_dim

        self.vn = vn
        
        self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, emb_dim),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Linear(emb_dim, emb_dim),
                nn.LeakyReLU(negative_slope = 0.1)
        )

        self.net_encoder = nn.Sequential(
                nn.Linear(net_dim, emb_dim),
                nn.LeakyReLU(negative_slope = 0.1)
        )
                
        self.convs = nn.ModuleList()

        if self.vn:
            self.virtualnode_embedding = nn.Embedding(1, emb_dim)   
            nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = nn.ModuleList()

            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(
                        nn.Sequential(
                            nn.Linear(emb_dim, emb_dim), 
                            nn.LeakyReLU(negative_slope = 0.1),
                            nn.Linear(emb_dim, emb_dim),
                            nn.LeakyReLU(negative_slope = 0.1)
                        )
                )

        for layer in range(num_layer):
            self.convs.append(HyperConvLayer(emb_dim, emb_dim))

        self.fc1_node = nn.Linear((self.num_layer + 1) * emb_dim, 256)
        self.fc2_node = nn.Linear(256, self.out_node_dim)

        self.fc1_net = nn.Linear((self.num_layer + 1) * emb_dim, 64)
        self.fc2_net = nn.Linear(64, self.out_net_dim)

        self.final_node_mlp = nn.Linear(self.out_node_dim, 1)


    def forward(self, data, device):
        node_features, net_features, edge_index_sink_to_net, edge_weight_sink_to_net, edge_index_source_to_net, batch, num_vn = data['node'].x.to(device), data['net'].x.to(device), data['node', 'as_a_sink_of', 'net'].edge_index, data['node', 'as_a_sink_of', 'net'].edge_weight, data['node', 'as_a_source_of', 'net'].edge_index.to(device), data.batch.to(device), data.num_vn

        # Masking
        edge_index_sink_to_net, edge_mask = dropout_edge(edge_index_sink_to_net, p = 0.4)
        edge_index_sink_to_net = edge_index_sink_to_net.to(device)
        edge_weight_sink_to_net = edge_weight_sink_to_net[edge_mask].to(device)
        
        h_list = [self.node_encoder(node_features)]
        h_net_list = [self.net_encoder(net_features)]

        if self.vn:
            virtualnode_embedding = self.virtualnode_embedding(torch.zeros(num_vn).to(batch.dtype).to(batch.device))

        for layer in range(self.num_layer):
            # Aggregate virutal node embedding
            if self.vn:
                h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            
            h_inst, h_net = self.convs[layer](h_list[layer], h_net_list[layer], edge_index_source_to_net, edge_index_sink_to_net, edge_weight_sink_to_net)
            h_list.append(h_inst)
            h_net_list.append(h_net)

            if (layer < self.num_layer - 1) and self.vn:
                virtualnode_embedding_temp = global_mean_pool(h_list[layer], batch) + virtualnode_embedding #global_mean_pool(h_list[layer], batch)
                virtualnode_embedding = virtualnode_embedding + self.mlp_virtualnode_list[layer](virtualnode_embedding_temp)
        
        node_representation = torch.cat(h_list, dim = 1)
        net_representation = torch.cat(h_net_list, dim = 1)

        node_representation = self.final_node_mlp(F.leaky_relu(self.fc2_node(F.leaky_relu(self.fc1_node(node_representation), negative_slope = 0.1)), negative_slope = 0.1))
        net_representation = torch.abs(F.leaky_relu(self.fc2_net(F.leaky_relu(self.fc1_net(net_representation), negative_slope = 0.1)), negative_slope = 0.1))

        return node_representation, net_representation
        