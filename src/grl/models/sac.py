import stable_baselines3
import torch as th
from stable_baselines3.common.utils import get_device
from torch_geometric.nn.models.basic_gnn import GCN, GAT, GraphSAGE

from environment.reward.res_penalty_reward import RESPenaltyReward
from environment.utils import env_graph
from gnn.graph_extractor import GraphExtractor
from .base_model import BaseModel


class SAC(BaseModel):

    def __init__(
        self,
        seed: int,
        name: str,
        verbose: int,
        train_episodes: int,
        eval_episodes: int,
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
        }

        for key, error_message in required_keys.items():
            if key not in model_params or model_params[key] is None:
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
        self.device = get_device("auto")

        super().__init__(
            seed=seed,
            name=name,
            verbose=verbose,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            model_params=model_params_cp,
            env_params=env_params,
        )


class GCN_SAC(SAC):
    def __init__(
        self,
        seed: int,
        name: str,
        verbose: int,
        train_episodes: int,
        eval_episodes: int,
        model_params: dict,
        env_params: dict,
        gnn_params: dict,
    ):
        super().__init__(
            seed=seed,
            name=name,
            verbose=verbose,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
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
        del gnn_params["class"]
        self._env_kwargs["gnn"] = GCN(**gnn_params)
        self._model_params["policy_kwargs"]["features_extractor_class"] = GraphExtractor


class GAT_SAC(SAC):
    def __init__(self, path, seed, name="GAT-SAC"):

        super().__init__(path, seed, name)

    def _init_env(self, name, config):
        print("Creating Graph env - " + name)

        gnn = GAT(
            in_channels=config["gnn_in_channels"],
            hidden_channels=config["gnn_hidden_channels"],
            out_channels=config["gnn_out_channels"],
            num_layers=config["gnn_num_layers"],
            dropout=config["gnn_dropout"],
            heads=config["gnn_heads"],
            v2=config["gnn_gatv2"],
            concat=config["gnn_concat"],
        ).to(device=self.device)

        reward = RESPenaltyReward(res_bonus=config["res_bonus"])

        gnn = th.compile(gnn)

        env = env_graph(name, reward, self.seed, gnn)
        return env

    def _setup_train(self, env, config):
        extractor = GraphExtractor
        extractor_kwargs = dict(
            ignored_keys=["step", "gen_p", "gen_p_before_curtail", "line_status"]
        )

        net_arch = [config["num_units_layer"]] * config[
            "num_hidden_layers"
        ]  # Network Architecture
        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=config["activation_fn"],
            optimizer_class=config["optimizer"],
            features_extractor_class=extractor,
            features_extractor_kwargs=extractor_kwargs,
        )

        sac_model = self.model(
            "MultiInputPolicy",
            env,
            learning_rate=config["learning_rate"],
            verbose=self.verbose,
            gamma=config["gamma"],
            use_sde=config["use_sde"],
            ent_coef=config["ent_coef"],
            policy_kwargs=policy_kwargs,
            sde_sample_freq=config["sde_sample_freq"],
            device=self.device,
            seed=self.seed,
            gradient_steps=config["gradient_steps"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            tau=config["tau"],
            target_update_interval=config["target_update_interval"],
            learning_starts=config["learning_starts"],
            train_freq=config["train_freq"],
            action_noise=config["action_noise"],
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            target_entropy=config["target_entropy"],
            use_sde_at_warmup=config["use_sde_at_warmup"],
        )
        return sac_model


class SAGE_SAC(SAC):
    def __init__(self, path, seed, name="SAGE-SAC"):

        super().__init__(path, seed, name)

    def _init_env(self, name, config):
        print("Creating Graph env - " + name)

        gnn = GraphSAGE(
            in_channels=config["gnn_in_channels"],
            hidden_channels=config["gnn_hidden_channels"],
            out_channels=config["gnn_out_channels"],
            num_layers=config["gnn_num_layers"],
            dropout=config["gnn_dropout"],
        ).to(device=self.device)

        reward = RESPenaltyReward(res_bonus=config["res_bonus"])

        gnn = th.compile(gnn)

        env = env_graph(name, reward, self.seed, gnn)
        return env

    def _setup_train(self, env, config):
        extractor = GraphExtractor
        extractor_kwargs = dict(
            ignored_keys=["step", "gen_p", "gen_p_before_curtail", "line_status"]
        )

        net_arch = [config["num_units_layer"]] * config[
            "num_hidden_layers"
        ]  # Network Architecture
        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=config["activation_fn"],
            optimizer_class=config["optimizer"],
            features_extractor_class=extractor,
            features_extractor_kwargs=extractor_kwargs,
        )

        sac_model = self.model(
            "MultiInputPolicy",
            env,
            learning_rate=config["learning_rate"],
            verbose=self.verbose,
            gamma=config["gamma"],
            use_sde=config["use_sde"],
            ent_coef=config["ent_coef"],
            policy_kwargs=policy_kwargs,
            sde_sample_freq=config["sde_sample_freq"],
            device=self.device,
            seed=self.seed,
            gradient_steps=config["gradient_steps"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            tau=config["tau"],
            target_update_interval=config["target_update_interval"],
            learning_starts=config["learning_starts"],
            train_freq=config["train_freq"],
            action_noise=config["action_noise"],
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            target_entropy=config["target_entropy"],
            use_sde_at_warmup=config["use_sde_at_warmup"],
        )
        return sac_model
