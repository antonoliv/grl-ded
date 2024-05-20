import numpy as np

from stable_baselines3.sac.policies import SACPolicy
from .networks.gcn import GCN
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import  Schedule
from torch_geometric.data import Data
class GCNExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Dict,
            gnn_arch: Tuple[int, ...] = (50, 15),
            dropout: float = 0.5,
    ):

        assert("x" in observation_space.spaces and len(observation_space["x"].shape) == 2), ("Invalid Feature Matrix")
        self.n_nodes = observation_space["x"].shape[0]

        in_features = 5

        assert ("edge_idx" in observation_space.spaces and
                observation_space["edge_idx"].shape[0] == 2), ("Invalid Graph Edge Index")
        self.n_edges = observation_space["edge_idx"].shape[1]


        assert ("edge_weight" in observation_space.spaces and
                len(observation_space["edge_weight"].shape) == 1 and
                observation_space["edge_weight"].shape[0] == self.n_edges), ("Invalid Graph Edge Weight")

        gnn_arch = (in_features,) + gnn_arch

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

        self.gcn = GCN(gnn_arch).to(device=th.device("cuda"))
        self._features_dim = features_dim
        self._dropout = dropout

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        data_batch = []
        batch_size = observations['x'].shape[0]
        x_b = []

        for i in range(batch_size):

            edge_idx, edge_weight = self._get_edge_attr(observations['edge_idx'][i], observations['edge_weight'][i])
            if edge_idx is None or edge_weight is None:
                x_b.append(th.zeros(size=(self.n_nodes, self._out_features)).to(device=th.device("cuda"), dtype=th.float32))
            else:
                data = Data(x=observations['x'][i].to(dtype=th.float32),
                            edge_index=edge_idx.to(device=th.device("cuda"), dtype=th.int64),
                            edge_attr=edge_weight.to(device=th.device("cuda"), dtype=th.float32))
                x_b.append(self.gcn.forward(data, self._dropout))

        # for i in range(batch_size):
        #
        #     edge_idx, edge_weight = self._get_edge_attr(observations['edge_idx'][i], observations['edge_weight'][i])
        #     data = Data(x=observations['x'][i].to(device=th.device("cuda"), dtype=th.float32),
        #                     edge_index=edge_idx.to(device=th.device("cuda"), dtype=th.int64),
        #                     edge_attr=edge_weight.to(device=th.device("cuda"), dtype=th.float32))
        #     data_batch.append(data)
        #
        # x_b = self.gcn.forward(data_batch, self._dropout)

        if len(x_b) == 0:
            return th.zeros(size=(batch_size, self._features_dim)).to(device=th.device("cuda"), dtype=th.float32)

        x = th.stack(x_b, dim=0)

        flatten = nn.Flatten()

        flattened_spaces= []

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
            edge_idx = edge_idx[:, :i+1]
            edge_weight = edge_weight[:i+1]
        else:
            edge_idx = None
            edge_weight = None

        return edge_idx, edge_weight




