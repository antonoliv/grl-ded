import copy

import grid2op
import numpy as np
import torch
import torch_geometric
from gymnasium import spaces
from stable_baselines3.common.utils import get_device


class GraphObservationSpace(spaces.Dict):
    """
    Graph Observation Space for the environment

    Main features:
    - Handle the graph observation space
    - Directly extract features with GNN
    - Handle scaled and unscaled observations
    - Handle step and time features
    """

    def __init__(
        self,
        init_env: grid2op.Environment,
        gnn: torch_geometric.nn.models,
        scaled: bool,
        step: bool,
    ):
        """
        Init function for the GraphObservationSpace class.

        :param init_env:    grid2op observation environment
        :param gnn:         Graph Neural Network
        :param scaled:      if true observations are scaled
        :param step:        if true step features are used instead of time features
        """

        if init_env is None:
            raise ValueError("Environment is not initialized")
        if gnn is None:
            raise ValueError("GNN is not initialized")
        if scaled is None:
            raise ValueError("Scaled is not initialized")
        if step is None:
            raise ValueError("Step is not initialized")

        # Initialize parameters
        self.device = get_device("auto")
        self._init_env = init_env
        self.gnn = gnn.to(self.device)
        self.step = step
        self.scaled = scaled

        self.n_nodes = init_env.n_sub
        self.n_edges = init_env.n_line

        self.high_p = self._get_node_pmax()
        self.low_p = self._get_node_pmin()
        self.high_res = self._get_node_resmax()

        if self.step:
            # if step is used, the input features are 6
            self._in_features = 6
        else:
            # else, time features (day_of_the_week, hour_of_day, minute_of_hour) are used
            self._in_features = 8

        # Check if the input features of the GNN are correct
        if gnn.in_channels != self._in_features:
            raise ValueError(
                f"Expected input features to be {self._in_features} but got {gnn.in_channels}"
            )

        self._out_features = gnn.out_channels

        extra_losses = 0.3 * np.abs(init_env.gen_pmax).sum()

        # Step space
        step = spaces.Box(
            np.full(shape=(1,), fill_value=0, dtype=np.int64),
            np.full(shape=(1,), fill_value=2016, dtype=np.int64),
            (1,),
            np.int64,
        )

        # Feature Matrix space
        x_space = spaces.Box(
            np.full(
                shape=(self.n_nodes, self._out_features),
                fill_value=-np.inf,
                dtype=np.float32,
            ),
            np.full(
                shape=(self.n_nodes, self._out_features),
                fill_value=+np.inf,
                dtype=np.float32,
            ),
            (self.n_nodes, self._out_features),
            np.float32,
        )

        # gen_p space
        gen_p_space = spaces.Box(
            np.full(shape=(init_env.n_gen,), fill_value=0.0, dtype=np.float32)
            - extra_losses
            + init_env._tol_poly,
            init_env.gen_pmax + extra_losses + init_env._tol_poly,
            (init_env.n_gen,),
            np.float32,
        )

        # line_status space
        line_status_space = spaces.Box(
            np.full(shape=(init_env.n_line,), fill_value=0.0, dtype=np.int32),
            np.full(shape=(init_env.n_line,), fill_value=1.0, dtype=np.int32),
            (init_env.n_line,),
            np.int32,
        )

        # Initialize the observation space
        spaces.Dict.__init__(
            self,
            spaces=dict(
                {
                    "x": x_space,  # Feature Matrix ({load_p, load_q, nres_p, res_p, res_p_before_curtail, step})
                    "line_status": line_status_space,  # Line status (0: disconnected, 1: connected)
                    "step": step,  # Current episode step
                    "gen_p": gen_p_space,  # Generation output of all generators
                    "gen_p_before_curtail": copy.deepcopy(gen_p_space),
                    # Generation of Renewable sources before curtail actions
                }
            ),
        )

    def _get_x(self, obs: grid2op.Observation) -> np.ndarray:
        """
        Get the feature matrix from current observation.

        :param obs: current observation
        :return: feature matrix
        """

        # Initialize x as a matrix of zeros
        x = np.full(
            shape=(self.n_nodes, self._in_features), fill_value=0, dtype=np.float32
        )

        # Initialize node_pmax, node_pmin, node_resmax
        node_pmax = self.high_p.copy()
        node_pmin = self.low_p.copy()
        node_resmax = self.high_res.copy()

        # Aggregate load attributes
        for load in range(obs.n_load):
            node_id = obs.load_to_subid[load]
            x[node_id][0] = obs.load_p[load]
            x[node_id][1] = obs.load_q[load]

        # Aggregate generation attributes
        for gen in range(obs.n_gen):
            node_id = obs.gen_to_subid[gen]

            if obs.gen_renewable[gen]:
                # if generator is renewable add the generation output to the feature matrix
                x[node_id][3] += obs.gen_p[gen]
                x[node_id][4] += obs.gen_p_before_curtail[gen]

                # Update node_pmax
                # node_pmax[node_id] += obs.gen_p_before_curtail[gen]
            else:
                x[node_id][2] += obs.gen_p[gen]

        for node in range(self.n_nodes):
            if self.scaled:
                # if scaled, normalize the features

                # if node_pmax - node_pmin > 0, normalize nres_p, else set the feature to -1
                if (node_pmax[node] - node_pmin[node]) > 0:
                    x[node][2] = np.interp(
                        x[node][2], [node_pmin[node], node_pmax[node]], [-1, 1]
                    )
                    # x[node][2] = -1 + 2 * (x[node][2] - node_pmin[node]) / (node_pmax[node] - node_pmin[node])
                else:
                    x[node][2] = -1

                # if node has renewable power normalize features, else set the features to -1
                if node_resmax[node] > 0:

                    # if res_p_before_curtail > 0, normalize res_p, else set the feature to -1
                    if x[node][4] > 0:
                        x[node][3] = np.interp(x[node][3], [0, x[node][4]], [-1, 1])
                        # x[node][3] = -1 + 2 * (x[node][3] / x[node][4])
                    else:
                        x[node][3] = -1

                    # normalize res_p
                    x[node][4] = np.interp(x[node][4], [0, node_resmax[node]], [-1, 1])
                    # x[node][4] = -1 + 2 * (x[node][4] / node_resmax[gen])
                else:
                    x[node][4] = -1
                    x[node][3] = -1

            # if step is used, set the step feature, else set the time features
            if self.step:
                x[node][5] = obs.current_step
            else:
                x[node][5] = obs.day_of_week
                x[node][6] = obs.hour_of_day
                x[node][7] = obs.minute_of_hour

        return x

    def _get_graph(
        self, observation: grid2op.Observation
    ) -> (np.ndarray, np.ndarray, int):
        """
        Get the graph attributes from the current observation.

        :param observation: grid2op observation
        :return: edge_idx, edge_weight, n_active_edges
        """

        # Initialize variables
        edge_idx = [[], []]  # edge_idx = [[or1, or2, ..., orn], [ex1, ex2, ..., exn]]
        edge_weight = []  # edge_weight = [rho1, rho2, ..., rhon]
        n_active_edges = 0  # n_active_edges

        for line in range(self.n_edges):
            node1 = observation.line_or_to_subid[line]
            node2 = observation.line_ex_to_subid[line]

            if observation.line_status[line]:
                # if line connected update variables
                edge_idx[0].append(node1)
                edge_idx[1].append(node2)
                edge_weight.append(observation.rho[line])
                n_active_edges += 1

        return (
            np.array(edge_idx, dtype=np.int32),
            np.array(edge_weight, dtype=np.float32),
            n_active_edges,
        )

    def to_gym(self, observation: grid2op.Observation) -> dict:
        """
        Convert grid2op observation to gym observation.

        :param observation: grid2op observation
        :return: gym observation
        """

        # Get graph attributes
        edge_idx, edge_weight, n_active_edges = self._get_graph(observation)

        if n_active_edges > 0:
            # if there is active edges, get the feature matrix and pass it through the GNN
            x = self._get_x(observation)
            x = self.gnn.forward(
                torch.tensor(x, device=self.device, dtype=torch.float32),
                torch.tensor(edge_idx, device=self.device, dtype=torch.int64),
                torch.tensor(edge_weight, device=self.device, dtype=torch.float32),
            )
        else:
            # else set the feature matrix to zeros
            x = torch.zeros((self.n_nodes, self._out_features), device=self.device)

        return {
            "x": x.detach().cpu().numpy(),
            "line_status": observation.line_status,
            "step": np.array([observation.current_step]),
            "gen_p": observation.gen_p,
            "gen_p_before_curtail": observation.gen_p_before_curtail,
        }

    def _get_node_pmax(self) -> np.ndarray:
        """
        Get the maximum non-renewable power output of each node.

        :return: maximum non-renewable power output of each node
        """

        node_max = np.zeros(shape=(self.n_nodes,), dtype=np.float32)
        for gen in range(self._init_env.n_gen):
            node_id = self._init_env.gen_to_subid[gen]

            if self._init_env.gen_redispatchable[gen]:
                node_max[node_id] += self._init_env.gen_pmax[gen]

        return node_max

    def _get_node_pmin(self) -> np.ndarray:
        """
        Get the minimum non-renewable power output of each node.

        :return: minimum non-renewable power output of each node
        """

        node_min = np.zeros(shape=(self.n_nodes,), dtype=np.float32)
        for gen in range(self._init_env.n_gen):
            node_id = self._init_env.gen_to_subid[gen]
            if self._init_env.gen_redispatchable[gen]:
                node_min[node_id] += self._init_env.gen_pmin[gen]

        return node_min

    def _get_node_resmax(self) -> np.ndarray:
        """
        Get the maximum renewable power output of each node.

        :return: maximum renewable power output of each node
        """

        node_max = np.zeros(shape=(self.n_nodes,), dtype=np.float32)
        for gen in range(self._init_env.n_gen):
            node_id = self._init_env.gen_to_subid[gen]

            if self._init_env.gen_renewable[gen]:
                node_max[node_id] += self._init_env.gen_pmax[gen]

        return node_max

    def __getstate__(self):
        """
        Get state, required for compatibility with stable-baselines3.

        :return: state
        """
        state = self.__dict__.copy()
        if "gnn" in state:
            del state["gnn"]

        return state

    def close(self):
        """
        Get state, required for compatibility with stable-baselines3.

        :return: state
        """

        if hasattr(self, "gnn"):
            self.gnn = None

        if hasattr(self, "_init_env"):
            self._init_env = None
