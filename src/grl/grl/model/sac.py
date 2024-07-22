import stable_baselines3
from torch_geometric.nn.models.basic_gnn import GCN, GAT, GraphSAGE
from torch_geometric.seed import seed_everything

from .common import BaseModel, GraphExtractor


class SAC(BaseModel):

    def __init__(
        self,
        seed: int,
        name: str,
        verbose: int,
        model_params: dict,
        env_params: dict,
    ):

        required_keys = {
            "learning_rate": "Learning Rate not defined",
            "gamma": "Discount Factor not defined",
            "ent_coef": "Entropy Coefficient not defined",
            "gradient_steps": "Gradient Steps not defined",
            "buffer_size": "Buffer Size not defined",
            "batch_size": "Batch Size not defined",
            "tau": "Tau not defined",
            "target_update_interval": "Target Update Interval not defined",
            "learning_starts": "Learning Starts not defined",
            "train_freq": "Train Frequency not defined",
            "target_entropy": "Target Entropy not defined",
            "use_sde": "Use SDE not defined",
            "sde_sample_freq": "SDE Sample Frequency not defined",
            "use_sde_at_warmup": "Use SDE at Warmup not defined",
            "optimizer": "Optimizer not defined",
            "activation_fn": "Activation Function not defined",
            "num_units_layer": "Number of Units per Layer not defined",
            "num_hidden_layers": "Number of Hidden Layers not defined",
            "action_noise": "Action Noise not defined",
        }

        for key, error_message in required_keys.items():
            if key not in model_params:
                raise ValueError(error_message)

        net_arch = [model_params["num_units_layer"]] * model_params["num_hidden_layers"]

        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=model_params["activation_fn"],
            optimizer_class=model_params["optimizer"],
        )

        model_params_cp = model_params.copy()
        model_params_cp["policy_kwargs"] = policy_kwargs
        del model_params_cp["num_units_layer"]
        del model_params_cp["num_hidden_layers"]
        del model_params_cp["optimizer"]
        del model_params_cp["activation_fn"]

        model_params_cp["class"] = stable_baselines3.SAC

        super().__init__(
            seed=seed,
            name=name,
            verbose=verbose,
            model_params=model_params_cp,
            env_params=env_params,
        )

        self._env_kwargs["obs_graph"] = False



class GCN_SAC(SAC):
    def __init__(
        self,
        seed: int,
        name: str,
        verbose: int,
        model_params: dict,
        env_params: dict,
        gnn_params: dict,
    ):
        #
        # required_keys = {
        #     "in_channels": "Input Channels not defined",
        #     "hidden_channels": "Hidden Channels not defined",
        #     "num_layers": "Number of Layers not defined",
        #     "out_channels": "Output Channels not defined",
        #     "dropout": "Dropout not defined",
        #     "act": "Activation Function not defined",
        #     "act_first": "Activation First not defined",
        #     "act_kwargs": "Activation Function Arguments not defined",
        #     "norm": "Normalization not defined",
        #     "norm_kwargs": "Normalization Arguments not defined",
        #     "jk": "Jumping Knowledge not defined",
        #     "aggr": "Aggregation Function not defined",
        #     "aggr_kwargs": "Aggregation Arguments not defined",
        #     "flow": "Flow Direction not defined",
        #     "node_dim": "Node Dimension not defined",
        #     "decomposed_layers": "Decomposed Layers not defined",
        # }
        #
        # for key, error_message in required_keys.items():
        #     if key not in gnn_params:
        #         raise ValueError(error_message)

        super().__init__(
            seed=seed,
            name=name,
            verbose=verbose,
            model_params=model_params,
            env_params=env_params,
        )

        self._env_kwargs["obs_graph"] = True

        if gnn_params["in_channels"] is None:
            raise ValueError("Input Channels is empty")
        if gnn_params["hidden_channels"] is None:
            raise ValueError("Hidden Channels is empty")
        if gnn_params["num_layers"] is None:
            raise ValueError("Number of Layers is empty")

        gnn_params = gnn_params.copy()

        seed_everything(seed)
        self._env_kwargs["gnn"] = GCN(**gnn_params)
        self._model_params["policy_kwargs"]["features_extractor_class"] = GraphExtractor



class GAT_SAC(SAC):
    def __init__(
        self,
        seed: int,
        name: str,
        verbose: int,
        model_params: dict,
        env_params: dict,
        gnn_params: dict,
    ):
        super().__init__(
            seed=seed,
            name=name,
            verbose=verbose,
            model_params=model_params,
            env_params=env_params,
        )

        self._env_kwargs["obs_graph"] = True

        if gnn_params["in_channels"] is None:
            raise ValueError("Input Channels is empty")
        if gnn_params["hidden_channels"] is None:
            raise ValueError("Hidden Channels is empty")
        if gnn_params["num_layers"] is None:
            raise ValueError("Number of Layers is empty")

        gnn_params = gnn_params.copy()

        seed_everything(seed)
        self._env_kwargs["gnn"] = GAT(**gnn_params)
        self._model_params["policy_kwargs"]["features_extractor_class"] = GraphExtractor


class SAGE_SAC(SAC):
    def __init__(
        self,
        seed: int,
        name: str,
        verbose: int,
        model_params: dict,
        env_params: dict,
        gnn_params: dict,
    ):
        super().__init__(
            seed=seed,
            name=name,
            verbose=verbose,
            model_params=model_params,
            env_params=env_params,
        )

        self._env_kwargs["obs_graph"] = True

        if gnn_params["in_channels"] is None:
            raise ValueError("Input Channels is empty")
        if gnn_params["hidden_channels"] is None:
            raise ValueError("Hidden Channels is empty")
        if gnn_params["num_layers"] is None:
            raise ValueError("Number of Layers is empty")

        gnn_params = gnn_params.copy()

        seed_everything(seed)
        self._env_kwargs["gnn"] = GraphSAGE(**gnn_params)
        self._model_params["policy_kwargs"]["features_extractor_class"] = GraphExtractor
