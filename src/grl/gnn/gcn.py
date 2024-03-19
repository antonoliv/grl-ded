class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(5, 1000, improved=True)
        self.conv2 = GCNConv(1000, 1000, improved=True)
        self.fully_con1 = torch.nn.Linear(1000, 1)

    def forward(self, data, prob, batch=None):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=prob)
        x = self.fully_con1(x)
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
