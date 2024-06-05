import torch

obs_keys = [
    'gen_p',
    # 'gen_q',
    # 'gen_theta',
    # 'gen_v',

    'gen_p_before_curtail',

    'load_p',
    'load_q',
    # 'load_theta',
    # 'load_v',

    'line_status',
    'rho'
]

storage_keys = [
    'storage_charge',
    'storage_power',
    'storage_power_target',
    'storage_theta'
]

delta_time = [
    'delta_time',
]

time_keys = [
    'minute_of_hour',
    'hour_of_day',
    'day',
    'day_of_week',
    'month',
    'year',
]

timestep_overflow_keys = [
    "timestep_overflow"
]

maintenance_keys = [
    'duration_next_maintenance',
    'time_next_maintenance',

    'time_before_cooldown_line',
    'time_before_cooldown_sub',
]


def get_obs_keys(storage: bool, maintenance: bool, delta_time: bool, time: bool, timestep_overflow: bool):
    ret = list(obs_keys)
    if storage:
        ret.extend(storage_keys)

    if delta_time:
        ret.extend(time_keys)

    if time:
        ret.extend(time_keys)

    if maintenance:
        ret.extend(maintenance_keys)

    if timestep_overflow:
        ret.extend(timestep_overflow_keys)

    return ret


import copy

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.utils import get_device


class GraphObservationSpace(spaces.Dict):
    def __init__(self, init_space, gnn):
        # do as you please here



        self.n_nodes = init_space.n_sub
        self.n_edges = init_space.n_line

        self._in_features = 5
        self._out_features = 15

        self.device = get_device("auto")

        # self.gcn = GCN(gnn_arch, self.device)

        import torch
        from torch_geometric.nn.models import GCN
        self.gnn = gnn

        step = spaces.Box(
            np.full(shape=(1,), fill_value=0, dtype=np.int64),
            np.full(shape=(1,), fill_value=2016, dtype=np.int64),
            (1,),
            np.int64,
        )

        # n_active_edge_space = spaces.Box(
        #     np.full(shape=(1,), fill_value=0, dtype=np.int64),
        #     np.full(shape=(1,), fill_value=self.n_edges, dtype=np.int64),
        #     (1,),
        #     np.int64,
        # )

        # edge_idx_space = spaces.Box(
        #     np.full(shape=(2, self.n_edges), fill_value=-1, dtype=np.int64),
        #     np.full(shape=(2, self.n_edges), fill_value=self.n_edges - 1, dtype=np.int64),
        #     (2, self.n_edges),
        #     np.int64
        # )
        #
        # edge_weight_space = spaces.Box(
        #     np.full(shape=(self.n_edges,), fill_value=-1, dtype=np.float32),
        #     np.full(shape=(self.n_edges,), fill_value=1, dtype=np.float32),
        #     (self.n_edges,),
        #     np.float32
        # )

        # edge_or_space = spaces.Sequence(spaces.Box(0, 1, dtype=np.int64), seed=0)
        # edge_ex_space = spaces.Sequence(spaces.Box(0, 1, dtype=np.int64), seed=0)
        # edge_weight_space = spaces.Sequence(spaces.Box(0, 1, dtype=np.float32), seed=0)

        # matrix_space = spaces.Box(
        #     np.full(shape=(self.n_nodes, self.n_nodes), fill_value=-1, dtype=np.float32),
        #     np.full(shape=(self.n_nodes, self.n_nodes), fill_value=1, dtype=np.float32),
        #     (self.n_nodes, self.n_nodes),
        #     np.float32
        # )

        x_space = spaces.Box(
            np.full(shape=(self.n_nodes, self._out_features), fill_value=-np.inf, dtype=np.float32),
            np.full(shape=(self.n_nodes, self._out_features), fill_value=+np.inf, dtype=np.float32),
            (self.n_nodes, self._out_features),
            np.float32
        )

        self._init_space = init_space
        gen_p_space = spaces.Box(
            np.full(shape=(init_space.n_gen,), fill_value=0.0, dtype=np.float32)
            - _compute_extra_power_for_losses(init_space) + (init_space.obs_env._tol_poly),
            init_space.gen_pmax + _compute_extra_power_for_losses(init_space) + (init_space.obs_env._tol_poly),
            (init_space.n_gen,),
            np.float32,
        )

        edge_idx_mask_space = spaces.MultiBinary((2, self.n_edges))
        edge_weight_mask_space = spaces.MultiBinary(self.n_edges)

        # don't forget to initialize the base class
        spaces.Dict.__init__(self, spaces=dict({
            "x": x_space,
            # "n_active_edges": n_active_edge_space,
            # "edge_idx": edge_idx_space,
            # 'edge_or': edge_or_space,
            # 'edge_ex': edge_ex_space,
            # "matrix": matrix_space,
            "step": step,
            "gen_p": gen_p_space,
            "gen_p_before_curtail": copy.deepcopy(gen_p_space),
        }))
        # eg. Box.__init__(self, low=..., high=..., dtype=float)

    def _get_x(self, obs):

        n_features = 5

        x = np.full(shape=(self.n_nodes, n_features), fill_value=0, dtype=np.float32)
        for load in range(obs.n_load):
            node_id = obs.load_to_subid[load]
            x[node_id][0] = obs.load_p[load]
            x[node_id][1] = obs.load_q[load]

        for gen in range(obs.n_gen):
            node_id = obs.gen_to_subid[gen]
            if obs.gen_renewable[gen]:
                x[node_id][3] += obs.gen_p[gen]
                x[node_id][4] += obs.gen_p_before_curtail[gen]
            else:
                x[node_id][2] += obs.gen_p[gen]

        return x

    # def _get_graph(self, observation):
    #     edge_idx = np.full(shape=(2, self.n_edges), fill_value=-1, dtype=np.int64)
    #     edge_weight = np.full(shape=(self.n_edges,), fill_value=-1, dtype=np.float32)
    #     i = 0
    #
    #     for line in range(self.n_edges):
    #         node1 = observation.line_or_to_subid[line]
    #         node2 = observation.line_ex_to_subid[line]
    #         if observation.line_status[line]:
    #             edge_idx[0][i] = node1
    #             edge_idx[1][i] = node2
    #             edge_weight[i] = observation.rho[line]
    #             i += 1
    #
    #     return edge_idx, edge_weight

    def _get_graph(self, observation):
        edge_idx = np.full(shape=(2, self.n_edges), fill_value=-1, dtype=np.int64)
        edge_weight = np.full(shape=(self.n_edges,), fill_value=-1, dtype=np.float32)

        # edge_or = []
        # edge_ex = []
        # edge_idx = [[], []]
        # edge_weight = []
        i = 0

        for line in range(self.n_edges):
            node1 = observation.line_or_to_subid[line]
            node2 = observation.line_ex_to_subid[line]
            if observation.line_status[line]:
                edge_idx[0][line] = node1
                edge_idx[1][line] = node2
                edge_weight[line] = observation.rho[line]
                # edge_idx[0].append(node1)
                # edge_idx[1].append(node2)
                # edge_weight.append(observation.rho[line])
                i += 1
            else:
                pass

        return edge_idx, edge_weight, i

    def _get_matrix(self, observation):

        weight_matrix = np.full(shape=(self.n_nodes, self.n_nodes), fill_value=-1, dtype=np.float32)

        for line in range(self.n_edges):
            node1 = observation.line_or_to_subid[line]
            node2 = observation.line_ex_to_subid[line]
            if observation.line_status[line] and line != 19:
                weight_matrix[node1][node2] = observation.rho[line]

        return weight_matrix

    def to_gym(self, observation):
        x = self._get_x(observation)

        edge_idx, edge_weight, n_active_edges = self._get_graph(observation)

        x = self.gnn.forward(torch.tensor(x, device=self.device), torch.tensor(edge_idx, device=self.device), torch.tensor(edge_weight, device=self.device))
        # matrix = self._get_matrix(observation)

        return {
            "x": x.detach().cpu().numpy(),
            # "n_active_edges": np.array([n_active_edges]),
            # "edge_idx": edge_idx,
            # "edge_or": edge_or,
            # "edge_ex": edge_ex,
            # "edge_weight": edge_weight,
            # "edge_idx_mask": np.array([observation.line_status, observation.line_status]),
            # "edge_weight_mask": observation.line_status,
            # "matrix": matrix,
            "step": np.array([observation.current_step]),
            "gen_p": observation.gen_p,
            "gen_p_before_curtail": observation.gen_p_before_curtail,
        }
        # eg. return np.concatenate((obs.gen_p * 0.1, np.sqrt(obs.load_p))

    def _get_x_space(self, init_space):

        extra_losses = _compute_extra_power_for_losses(init_space) + (init_space.obs_env._tol_poly)

        x_shape = (self.n_nodes, self.n_features)
        high = np.empty(x_shape, dtype=np.float32)
        for sub in range(self.n_nodes):
            high[sub] = [+np.inf, +np.inf, extra_losses, extra_losses, extra_losses]

        for gen in range(init_space.n_gen):
            node_id = init_space.gen_to_subid[gen]

            if init_space.gen_renewable[gen]:
                high[node_id][3] += init_space.gen_pmax[gen]
                high[node_id][4] += init_space.gen_pmax[gen]
            else:
                high[node_id][2] += init_space.gen_pmax[gen]

        x_space = spaces.Box(
            np.array([-np.inf, -np.inf, -extra_losses, -extra_losses, -extra_losses] * self.n_nodes,
                     dtype=np.float32).reshape(x_shape),
            high,
            x_shape,
            np.float32
        )

        return x_space

    def close(self):
        if hasattr(self, "_init_space"):
            self._init_space = None  # this doesn't own the environment

def _compute_extra_power_for_losses(gridobj):
    """
    to handle the "because of the power losses gen_pmin and gen_pmax can be slightly altered"
    """

    return 0.3 * np.abs(gridobj.gen_pmax).sum()
