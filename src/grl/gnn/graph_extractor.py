from typing import Dict, List

import torch as th
from gymnasium import spaces
from torch import nn
from torch_geometric.data import Data, Batch

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN
)
from stable_baselines3.common.utils import get_device
from .networks.gat import GAT
from .networks.gcn import GCN
from stable_baselines3.common.preprocessing import is_image_space


def _get_from_matrix(matrix):
    # Initialize edge tensors with zeros
    mask = matrix > -1
    edge_idx = th.nonzero(mask, as_tuple=False).t()
    edge_weight = matrix[mask]

    return edge_idx, edge_weight

class FilterExtractor(BaseFeaturesExtractor):
    """
        Combined features extractor for Dict observation spaces.
        Builds a features extractor for each key of the space. Input from each space
        is fed through a separate submodule (CNN or MLP, depending on input shape),
        the output features are concatenated and fed through additional MLP network ("combined").

        :param observation_space:
        :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
            256 to avoid exploding network sizes.
        :param normalized_image: Whether to assume that the image is already normalized
            or not (this disables dtype and bounds checks): when True, it only checks that
            the space is a Box and has 3 dimensions.
            Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
        """

    def __init__(
            self,
            observation_space: spaces.Dict,
            cnn_output_dim: int = 256,
            normalized_image: bool = False,
            ignored_keys: list = []
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if(key not in ignored_keys):
                if is_image_space(subspace, normalized_image=normalized_image):
                    extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                    total_concat_size += cnn_output_dim
                else:
                    # The observation key is a vector, flatten it if needed
                    extractors[key] = nn.Flatten()
                    total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:

        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

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
            if key not in ["x",
                           "edge_idx",
                           "edge_weight",
                           "n_active_edges",
                           "gen_p",
                           "gen_p_before_curtail",
                           "matrix"]:
                features_dim += get_flattened_obs_dim(space)

        super().__init__(observation_space, features_dim=features_dim)

        self.device = get_device("auto")

        # self.gcn = GCN(gnn_arch, self.device)

        import torch
        from torch_geometric.nn.models import GCN
        self.gcn = GCN(in_channels=5, hidden_channels=100, out_channels=15, num_layers=2, dropout=dropout).to(device=self.device)
        self.gcn = torch.compile(self.gcn)
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
        # for key in observations.keys():
        #     observations[key].to(device="cpu")

        # matrices = observations['matrix'].split(1, dim=0)
        # edge_idx_list, edge_weight_list = zip(*self._get_from_matrix_parallel(matrices))

        # idx_mask = observations['edge_idx_mask'].to(dtype=th.bool)
        # edge_idx_list = observations['edge_idx'][idx_mask]

        # print(edge_idx_list)

        n_valid = 0
        for i in range(batch_size):
            # edge_idx, edge_weight = self._get_edge_attr(observations['edge_idx'][i], observations['edge_weight'][i], observations['n_active_edges'][i])
            # edge_idx, edge_weight = self._get_edge_attr(observations['edge_idx'][i], observations['edge_weight'][i], 0)
            # edge_idx, edge_weight = _get_from_matrix(observations['matrix'][i])
            # edge_idx, edge_weight = edge_idx_list[i], edge_weight_list[i]
            n_active_edges = observations['n_active_edges'][i]


            if n_active_edges == 0:
                mask[i] = False
            else:
            # Filter the edge_idx and edge_weight arrays
                if self.n_edges != observations['n_active_edges'][i]:
                    edge_mask = observations["edge_weight_mask"][i].to(dtype=th.bool)
                    edge_idx = observations['edge_idx'][i][:, edge_mask]
                    # print(edge_idx)
                    # print(edge_idx_list[i])
                    edge_weight = observations['edge_weight'][i][edge_mask]
                else:
                    edge_idx = observations['edge_idx'][i]
                    edge_weight = observations['edge_weight'][i]
                data = Data(x=observations['x'][i],
                            edge_index=edge_idx.to(dtype=th.int64),
                            edge_attr=edge_weight)
                data_batch.append(data)
                # x_b.append(self.gcn.forward(observations['x'][i], edge_idx.to(dtype=th.int64), edge_weight, self._dropout))
                # n_valid += 1

        # if len(x_b) == 0:
        #     return th.zeros(size=(batch_size, self._out_features), dtype=th.float32, device=self.device)
        # x = th.zeros(size=(batch_size, self.n_nodes, self._out_features), dtype=th.float32, device=self.device)
        # x[mask] = th.stack(x_b, dim=0)

        if n_valid > 0:
            batch = Batch.from_data_list(data_batch)

            x = self.gcn.forward(x=batch.x, edge_index=batch.edge_index, edge_weight=batch.edge_weight, batch=batch, batch_size=1).view(n_valid, -1, self._out_features)
            x_unmasked = th.full(size=(batch_size, self.n_nodes, self._out_features), fill_value=0, dtype=th.float32, device=self.device)
            x_unmasked[mask] = x
        else:
            x_unmasked = th.full(size=(batch_size, self.n_nodes, self._out_features), fill_value=0, dtype=th.float32, device=self.device)


        flatten = nn.Flatten()
        flattened_spaces = [flatten(observations[key]) for key in observations if key not in [
            "x", "edge_idx", "edge_weight", "n_active_edges", "gen_p", "gen_p_before_curtail", "matrix"]]
        flattened_spaces.append(flatten(x_unmasked))

        return th.cat(flattened_spaces, dim=1)

    def _get_edge_attr(self, edge_idx, edge_weight, n_active_edges):
        # Initialize edge tensors with zeros


        # Create a mask to filter out -1 values

        # print(edge_idx)
        # print(edge_weight)
        # mask = (edge_idx[0] != -1) & (edge_idx[1] != -1) & (edge_weight != -1)
        #
        # # Filter the edge_idx and edge_weight arrays
        # filtered_edge_idx = edge_idx[:, mask]
        # filtered_edge_weight = edge_weight[mask]

        # print(n_active_edges[0])
        if self.n_edges == n_active_edges:
            return edge_idx, edge_weight

        if n_active_edges == 0:
            return None, None

        # Filter the edge_idx and edge_weight arrays
        filtered_edge_idx = edge_idx[:, :int(n_active_edges)]
        filtered_edge_weight = edge_weight[:int(n_active_edges)]

        return filtered_edge_idx, filtered_edge_weight



    # def _get_from_matrix_parallel(self, matrices):
    #     from multiprocessing import Pool
    #     with Pool(processes=6) as pool:
    #         results = pool.map(_get_from_matrix, matrices)
    #     return results




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
