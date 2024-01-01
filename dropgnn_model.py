import random
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
torch.set_printoptions(profile="full")

class DropGIN(nn.Module):
        def __init__(self, dataset, num_layers, num_features, use_aux_loss, num_runs, p, augmentation):
            super(DropGIN, self).__init__()

            dim = dataset.hidden_units

            self.num_layers = num_layers
            self.num_runs = num_runs
            self.p = p
            self.use_aux_loss = use_aux_loss
            self.random_value = random.uniform(0, 0.2)
            self.augmentation = augmentation
            self.Conv = GINConv

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
            
            if use_aux_loss:
                self.aux_fcs = nn.ModuleList()
                self.aux_fcs.append(nn.Linear(num_features, dataset.num_classes))
                for i in range(self.num_layers):
                    self.aux_fcs.append(nn.Linear(dim, dataset.num_classes))
        
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
            
            # Do runs in paralel, by repeating the graphs in the batch
            x = x.unsqueeze(0).expand(self.num_runs, -1, -1).clone()
            drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * self.p).bool()
            if self.augmentation == 'rrni':
                x[drop] = self.random_value
            else:
                x[drop] = 0.0
            del drop
            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, self.num_runs) + torch.arange(self.num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
            for i in range(self.num_layers):   
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(self.num_runs, -1, x.size(-1)))
            del run_edge_index

            out = None
            for i, x in enumerate(outs):
                x = x.mean(dim=0)
                x = self.fcs[i](x) # No dropout layer in these experiments
                if out is None:
                    out = x
                else:
                    out += x
            
            if self.use_aux_loss:
                aux_out = torch.zeros(self.num_runs, out.size(0), out.size(1), device=out.device)
                run_batch = batch.repeat(self.num_runs) + torch.arange(self.num_runs, device=edge_index.device).repeat_interleave(batch.size(0)) * (batch.max() + 1)
                for i, x in enumerate(outs):
                    x = x.view(self.num_runs, -1, x.size(-1))
                    x = self.aux_fcs[i](x) # No dropout layer in these experiments
                    aux_out += x

                return F.log_softmax(out, dim=-1), F.log_softmax(aux_out, dim=-1)
            else:
                return F.log_softmax(out, dim=-1), 0