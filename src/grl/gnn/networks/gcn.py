import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    r"""
    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. By default, self-loops will be added
            in case :obj:`normalize` is set to :obj:`True`, and not added
            otherwise. (default: :obj:`None`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on-the-fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
        aggr (str or [str] or Aggregation, optional): The aggregation scheme
            to use, *e.g.*, :obj:`"sum"` :obj:`"mean"`, :obj:`"min"`,
            :obj:`"max"` or :obj:`"mul"`.
            In addition, can be any
            :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
            that automatically resolves to it).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
            If set to :obj:`None`, the :class:`MessagePassing` instantiation is
            expected to implement its own aggregation logic via
            :meth:`aggregate`. (default: :obj:`"add"`)
        aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective aggregation function in case it gets automatically
            resolved. (default: :obj:`None`)
        flow (str, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
        decomposed_layers (int, optional): The number of feature decomposition
            layers, as introduced in the `"Optimizing Memory Efficiency of
            Graph Neural Networks on Edge Computing Platforms"
            <https://arxiv.org/abs/2104.03058>`_ paper.
            Feature decomposition reduces the peak memory usage by slicing
            the feature dimensions into separated feature decomposition layers
            during GNN aggregation.
            This method can accelerate GNN execution on CPU-based platforms
            (*e.g.*, 2-3x speedup on the
            :class:`~torch_geometric.datasets.Reddit` dataset) for common GNN
            models such as :class:`~torch_geometric.nn.models.GCN`,
            :class:`~torch_geometric.nn.models.GraphSAGE`,
            :class:`~torch_geometric.nn.models.GIN`, etc.
            However, this method is not applicable to all GNN operators
            available, in particular for operators in which message computation
            can not easily be decomposed, *e.g.* in attention-based GNNs.
            The selection of the optimal value of :obj:`decomposed_layers`
            depends both on the specific graph dataset and available hardware
            resources.
            A value of :obj:`2` is suitable in most cases.
            Although the peak memory usage is directly associated with the
            granularity of feature decomposition, the same is not necessarily
            true for execution speedups. (default: :obj:`1`)
    """

    def __init__(self, arch, device):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(len(arch) - 1):
            self.convs.append(GCNConv(in_channels=arch[i],
                                      out_channels=arch[i + 1],
                                      cached=False,
                                      add_self_loops=None,
                                      normalize=True,
                                      bias=True,
                                      improved=True,
                                      aggr="add",
                                      aggr_kwargs=None,
                                      flow="source_to_target",
                                      node_dim=-2,
                                      decomposed_layers=1,
                                      ))

        self.to(device)

    def forward(self, data, mask, prob=0.5) -> Tensor:

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        for layer in self.convs:
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        x = F.dropout(x, p=prob)

        return x

# class PolicyGCN(torch.nn.Module):
#     def __init__(self):
#         super(PolicyGCN, self).__init__()
#         self.conv1 = GCNConv(5, 1000, improved=True)
#         self.conv2 = GCNConv(1000, 1000, improved=True)
#         self.fully_con1 = torch.nn.Linear(1000, 1)
#
#     def forward(self, data, mask, batch=None):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#
#         x = self.conv1(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = F.dropout(x)
#         x = self.fully_con1(x)
#         x = torch.masked_select(x.view(-1), mask)
#         batch = torch.masked_select(batch, mask)
#         x = softmax(x, batch)
#         return x
#
#
# class ValueGCN(torch.nn.Module):
#     def __init__(self):
#         super(ValueGCN, self).__init__()
#         self.conv1 = GCNConv(5, 1000, improved=True)
#         self.conv2 = GCNConv(1000, 1000, improved=True)
#         self.fully_con1 = torch.nn.Linear(1000, 100)
#
#     def forward(self, data, mask, batch=None):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#
#         x = self.conv1(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = F.dropout(x)
#         x = self.fully_con1(x)
#         x = global_mean_pool(x, batch).mean(dim=1)
#         return x
