

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch_geometric.utils import (sort_edge_index,
                                   softmax)


class GCN(torch.nn.Module):
    def __init__(self, arch):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(len(arch) - 1):
                self.convs.append(GCNConv(arch[i], arch[i + 1], improved=True))

    def forward(self, data, prob=0.5) -> torch.Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        for layer in self.convs:
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        x = F.dropout(x, p=prob)

        return x



class PolicyGCN(torch.nn.Module):
    def __init__(self):
        super(PolicyGCN, self).__init__()
        self.conv1 = GCNConv(5, 1000, improved=True)
        self.conv2 = GCNConv(1000, 1000, improved=True)
        self.fully_con1 = torch.nn.Linear(1000, 1)

    def forward(self, data, mask, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fully_con1(x)
        x = torch.masked_select(x.view(-1), mask)
        batch = torch.masked_select(batch, mask)
        x = softmax(x, batch)
        return x


class ValueGCN(torch.nn.Module):
    def __init__(self):
        super(ValueGCN, self).__init__()
        self.conv1 = GCNConv(5, 1000, improved=True)
        self.conv2 = GCNConv(1000, 1000, improved=True)
        self.fully_con1 = torch.nn.Linear(1000, 100)

    def forward(self, data, mask, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fully_con1(x)
        x = global_mean_pool(x, batch).mean(dim=1)
        return x
