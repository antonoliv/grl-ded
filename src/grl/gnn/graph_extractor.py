from typing import Dict, List

import torch as th
from gymnasium import spaces
from torch import nn
from torch_geometric.data import Data, Batch

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.utils import get_device
from .networks.gat import GAT
from .networks.gcn import GCN


class GCNExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Dict,
            gnn_arch: List[int] = [50, 15],
            dropout: float = 0.5,
    ):

        assert ("x" in observation_space.spaces and len(observation_space["x"].shape) == 2), ("Invalid Feature Matrix")
        self.n_nodes = observation_space["x"].shape[0]

        self._in_features = 5

        assert ("edge_idx" in observation_space.spaces and
                observation_space["edge_idx"].shape[0] == 2), ("Invalid Graph Edge Index")
        self.n_edges = observation_space["edge_idx"].shape[1]

        assert ("edge_weight" in observation_space.spaces and
                len(observation_space["edge_weight"].shape) == 1 and
                observation_space["edge_weight"].shape[0] == self.n_edges), ("Invalid Graph Edge Weight")

        gnn_arch = [self._in_features] + gnn_arch

        self._out_features = gnn_arch[-1]

        features_dim = self.n_nodes * self._out_features

        for key, space in observation_space.spaces.items():
            if (key != "x" and
                    key != "edge_idx" and
                    key != "edge_weight" and
                    key != "gen_p" and
                    key != "gen_p_before_curtail"):
                features_dim += get_flattened_obs_dim(space)

        super().__init__(observation_space, features_dim=features_dim)

        self.device = get_device("auto")
        self.gcn = GCN(gnn_arch, self.device)
        self._features_dim = features_dim
        self._dropout = dropout

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        data_batch = []
        batch_size = observations['x'].shape[0]
        x_b = []

        # for i in range(batch_size):
        #
        #     edge_idx, edge_weight = self._get_edge_attr(observations['edge_idx'][i], observations['edge_weight'][i])
        #     if edge_idx is None or edge_weight is None:
        #         x_b.append(th.zeros(size=(self.n_nodes, self._out_features)).to(device=device, dtype=th.float32))
        #     else:
        #         data = Data(x=observations['x'][i].to(dtype=th.float32),
        #                     edge_index=edge_idx.to(device=device, dtype=th.int64),
        #                     edge_attr=edge_weight.to(device=device, dtype=th.float32))
        #         x_b.append(self.gcn.forward(data, self._dropout))
        #
        #
        # if len(x_b) == 0:
        #     return th.zeros(size=(batch_size, self._features_dim)).to(device=device, dtype=th.float32)
        # x = th.stack(x_b, dim=0)

        mask = th.full(size=(batch_size,), fill_value=True, dtype=th.bool)

        n_valid = 0
        for i in range(batch_size):
            edge_idx, edge_weight = self._get_edge_attr(observations['edge_idx'][i], observations['edge_weight'][i])
            # edge_idx, edge_weight = self._get_from_matrix(observations['matrix'][i])
            if edge_idx is None or edge_weight is None or edge_idx.numel() == 0 or edge_weight.numel() == 0:
                mask[i] = False
            else:
                data = Data(x=observations['x'][i],
                            edge_index=edge_idx.to(dtype=th.int64),
                            edge_attr=edge_weight)
                data_batch.append(data)
                n_valid += 1

        batch = Batch.from_data_list(data_batch)

        x = self.gcn.forward(batch, mask, self._dropout)
        x = x.view(n_valid, -1, self._out_features)

        x_unmasked = th.full(size=(batch_size, self.n_nodes, self._out_features), fill_value=0, dtype=th.float32).to(
            device=self.device, dtype=th.float32)
        x_unmasked[mask] = x

        flatten = nn.Flatten()

        flattened_spaces = []

        for key, _ in observations.items():
            if (key != "x" and
                    key != "edge_idx" and
                    key != "edge_weight" and
                    key != "gen_p" and
                    key != "gen_p_before_curtail"):
                flattened_spaces.append(flatten(observations[key]))

        flattened_spaces.append(flatten(x_unmasked))
        return th.cat(flattened_spaces, dim=1)

    def _get_edge_attr(self, edge_idx, edge_weight):
        # Initialize edge tensors with zeros
        i = edge_idx.shape[1] - 1

        while edge_idx[0][i] == -1 and i != 0:
            i -= 1

        if i != 0:
            edge_idx = edge_idx[:, :i + 1]
            edge_weight = edge_weight[:i + 1]
        else:
            edge_idx = None
            edge_weight = None

        return edge_idx, edge_weight

    def _get_from_matrix(self, matrix):
        # Initialize edge tensors with zeros
        edge_idx = (matrix > -1).nonzero(as_tuple=False).t().contiguous()
        edge_weight = matrix[matrix > -1]

        return edge_idx, edge_weight


class GATExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Dict,
            gat_arch: List[int] = [50, 15],
            dropout: float = 0.5,
            heads: int = 1,
    ):

        assert ("x" in observation_space.spaces and len(observation_space["x"].shape) == 2), ("Invalid Feature Matrix")
        self.n_nodes = observation_space["x"].shape[0]

        in_features = 5

        assert ("edge_idx" in observation_space.spaces and
                observation_space["edge_idx"].shape[0] == 2), ("Invalid Graph Edge Index")
        self.n_edges = observation_space["edge_idx"].shape[1]

        assert ("edge_weight" in observation_space.spaces and
                len(observation_space["edge_weight"].shape) == 1 and
                observation_space["edge_weight"].shape[0] == self.n_edges), ("Invalid Graph Edge Weight")

        gat_arch = [in_features] + gat_arch

        self._out_features = gat_arch[-1]

        features_dim = self.n_nodes * self._out_features

        for key, space in observation_space.spaces.items():
            if (key != "x" and
                    key != "edge_idx" and
                    key != "edge_weight" and
                    key != "gen_p" and
                    key != "gen_p_before_curtail"):
                features_dim += get_flattened_obs_dim(space)

        super().__init__(observation_space, features_dim=features_dim)

        self.device = get_device("auto")
        self.gat = GAT(gat_arch, heads, self.device)
        self._features_dim = features_dim
        self._dropout = dropout

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        data_batch = []
        batch_size = observations['x'].shape[0]
        x_b = []

        # for i in range(batch_size):
        #
        #     edge_idx, edge_weight = self._get_edge_attr(observations['edge_idx'][i], observations['edge_weight'][i])
        #     if edge_idx is None or edge_weight is None:
        #         x_b.append(th.zeros(size=(self.n_nodes, self._out_features)).to(device=device, dtype=th.float32))
        #     else:
        #         data = Data(x=observations['x'][i].to(dtype=th.float32),
        #                     edge_index=edge_idx.to(device=device, dtype=th.int64),
        #                     edge_attr=edge_weight.to(device=device, dtype=th.float32))
        #         x_b.append(self.gcn.forward(data, self._dropout))
        #
        #
        # if len(x_b) == 0:
        #     return th.zeros(size=(batch_size, self._features_dim)).to(device=device, dtype=th.float32)
        # x = th.stack(x_b, dim=0)

        mask = []
        for i in range(batch_size):
            edge_idx, edge_weight = self._get_edge_attr(observations['edge_idx'][i], observations['edge_weight'][i])
            if edge_idx is None or edge_weight is None:
                mask[i] = 1
            data = Data(x=observations['x'][i].to(device=self.device, dtype=th.float32),
                        edge_index=edge_idx.to(device=self.device, dtype=th.int64),
                        edge_attr=edge_weight.to(device=self.device, dtype=th.float32))
            data_batch.append(data)

        batch = Batch.from_data_list(data_batch)
        x = self.gat.forward(batch, self._dropout)

        x = x.view(batch_size, -1, self._out_features)

        flatten = nn.Flatten()

        flattened_spaces = []

        for key, _ in observations.items():
            if (key != "x" and
                    key != "edge_idx" and
                    key != "edge_weight" and
                    key != "gen_p" and
                    key != "gen_p_before_curtail"):
                flattened_spaces.append(flatten(observations[key]))

        flattened_spaces.append(flatten(x))
        return th.cat(flattened_spaces, dim=1)

    def _get_edge_attr(self, edge_idx, edge_weight):
        # Initialize edge tensors with zeros
        i = edge_idx.shape[1] - 1

        while edge_idx[0][i] == -1 and i != 0:
            i -= 1

        if i != 0:
            edge_idx = edge_idx[:, :i + 1]
            edge_weight = edge_weight[:i + 1]
        else:
            edge_idx = th.zeros(size=(2, 1))
            edge_weight = th.zeros(size=(1,))

        return edge_idx, edge_weight
