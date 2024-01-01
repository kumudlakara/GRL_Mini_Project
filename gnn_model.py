# This implementation is based on https://github.com/weihua916/powerful-gnns, https://github.com/chrsmrrs/k-gnn/tree/master/examples and https://github.com/KarolisMart/DropGNN
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
torch.set_printoptions(profile="full")

class GIN(nn.Module):
        def __init__(self, dataset, num_features, num_layers, augmentation, p):
            super(GIN, self).__init__()

            dim = dataset.hidden_units

            self.num_layers = num_layers
            self.augmentation = augmentation
            self.Conv = GINConv
            self.p = p

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(self.Conv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers-1):
                self.convs.append(self.Conv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))
        
        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, self.Conv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            if self.augmentation == 'random':
                x = torch.cat([x, torch.randint(0, 100, (x.size(0), 1), device=x.device) / 100.0], dim=1)
            elif self.augmentation == 'prob_rni':
                drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * self.p).bool()
                random_features = torch.randint(0,100, (x.size(0), 1), device=x.device)/100.0
                for idx in range(drop.shape[0]):
                    if not drop[idx]:
                        random_features[idx] = 0.0
                x = torch.cat([x, random_features], dim=1)

            outs = [x]
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x)
            
            out = None
            for i, x in enumerate(outs):
                x = self.fcs[i](x) # No dropout for these experiments
                if out is None:
                    out = x
                else:
                    out += x
            return F.log_softmax(out, dim=-1), 0

    
