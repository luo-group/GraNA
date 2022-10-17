import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, GINConv, SuperGATConv, to_hetero
from torch_geometric.nn.norm import BatchNorm, PairNorm


class GraNA(nn.Module):
    def __init__(self, metadata, in_dim=8717, split_number=6284, hidden_dim=256, num_layer=7, conv_type='GEN'):
        super().__init__()

        self.total_number = in_dim
        self.split_number = split_number

        self.linear = nn.Linear(in_dim, hidden_dim)
        self.encoder = DeeperGCN(hidden_dim=hidden_dim, num_layers=num_layer, conv_type=conv_type)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, PE):
        x['protein'] = F.leaky_relu(self.linear(x['protein'])) + PE
        x = self.encoder(x, edge_index)

        return x

    def decode(self, z):
        
        z = self.decoder(z)
        z = torch.sigmoid(z)

        return z


class DeeperGCN(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5, dropout=0, conv_type='GEN'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer in range(self.num_layers):

            if conv_type == 'GEN':
                gcn = GENConv(hidden_dim, hidden_dim, learn_t=True, learn_p=True, learn_msg_scale=True)
            elif conv_type == 'GEN-false':
                gcn = GENConv(hidden_dim, hidden_dim)
            elif conv_type == 'GIN':
                gcn = GINConv(nn.Lineard(hidden_dim, hidden_dim))
            elif conv_type == 'SuperGAT':
                gcn = SuperGATConv(hidden_dim, hidden_dim)
            
            self.gcns.append(gcn)
            self.norms.append(PairNorm())

    def forward(self, x, edge_index):

        h = x

        h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        for layer in range(1, self.num_layers):

            h1 = self.gcns[layer](h, edge_index)
            
            h2 = self.norms[layer](h1)

            h = F.relu(h2) + h
            
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h
