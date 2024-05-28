import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    r"""
    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities in case of a bipartite graph.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops (in case
            :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, arch, heads, device):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()

        for i in range(len(arch) - 1):
            self.convs.append(GATConv(in_channels=arch[i],
                                      out_channels=arch[i + 1] // heads,
                                      heads=heads,
                                      concat=True,
                                      negative_slope=0.2,
                                      dropout=0.0,
                                      add_self_loops=True,
                                      edge_dim=None,
                                      fill_value='mean',
                                      bias=True,
                                      aggr="add",
                                      aggr_kwargs=None,
                                      flow="source_to_target",
                                      decomposed_layers=1,
                                      ))

        self.to(device)

    def forward(self, data, prob=0.5) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for layer in self.convs:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
        x = F.dropout(x, p=prob, training=self.training)

        return x
