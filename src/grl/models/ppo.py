import stable_baselines3
import torch as th
from stable_baselines3.common.utils import get_device
from torch_geometric.nn.models.basic_gnn import GCN, GAT, GraphSAGE

from environment.reward.res_penalty_reward import RESPenaltyReward
from environment.utils import env_graph, env_multi
from gnn.graph_extractor import GraphExtractor
from .base_model import BaseModel


# class PPO(BaseModel):
#
#     def __init__(self, path, seed, name="PPO", extractor=None, extractor_kwargs=None):
#
#         self.name = name
#         self.path = path
#         self.seed = seed
#
#         super().__init__(path, seed, name)
#
#         if extractor is None:
#             extractor = CombinedExtractor
#             extractor_kwargs = None
#
#         # Policy Parameters
#         optimizer = th.optim.Adam  # Optimizer
#         activation_fn = th.nn.ReLU  # Activation Function
#         num_units_layer = 256
#         num_hidden_layers = 2
#
#         net_arch = [num_units_layer] * num_hidden_layers  # Network Architecture
#         self.policy_kwargs = dict(
#             net_arch=net_arch,
#             activation_fn=activation_fn,
#             optimizer_class=optimizer,
#             features_extractor_class=extractor,
#             features_extractor_kwargs=extractor_kwargs,
#         )
#
#         self.model = stable_baselines3.PPO
#         self.verbose = 0
#         self.device = get_device("auto")
#
#         self.gamma = 0.99  # Discount Factor
#         self.learning_rate = 3e-4  # Learning Rate
#         self.ent_coef = "auto"
#         self.use_sde = False
#         self.sde_sample_freq = -1
#
#         self.n_steps = 2048
#         self.n_epochs = 10
#         self.batch_size = 64
#         self.gae_lambda = 0.95
#         self.clip_range = 0.2
#         self.clip_range_vf = None
#         self.ent_coef = 0.0
#         self.vf_coef = 0.5
#         self.target_kl = None
#         self.max_grad_norm = 0.5
#         self.use_rms_prop = True
#         self.rms_prop_eps = 1e-5
#         self.normalize_advantage = True
#         self.rollout_buffer_kwargs = None
#         self.rollout_buffer_class = None
#
#     def _init_env(self, name):
#         obs_attr = [
#                     'gen_p',
#                     # 'gen_q',
#                     # 'gen_theta',
#                     # 'gen_v',
#
#                     'gen_p_before_curtail',
#
#                     'load_p',
#                     'load_q',
#                     # 'load_theta',
#                     # 'load_v',
#
#                     'line_status',
#                     'rho',
#                     'step'
#                     ]
#
#         print("Creating MultiInput env - " + name)
#         env = env_multi(name, obs_attr, self.reward, self.seed)
#         return env
#
#     def _setup_train(self, env):
#         ppo_model = self.model("MultiInputPolicy",
#                                       env,
#                                       learning_rate=self.learning_rate,
#                                       gamma=self.gamma,
#                                       n_steps=self.n_steps,
#                                       batch_size=self.batch_size,
#                                       n_epochs=self.n_epochs,
#                                       gae_lambda=self.gae_lambda,
#                                       clip_range=self.clip_range,
#                                       clip_range_vf=self.clip_range_vf,
#                                       normalize_advantage=self.normalize_advantage,
#                                       ent_coef=self.ent_coef,
#                                       vf_coef=self.vf_coef,
#                                       max_grad_norm=self.max_grad_norm,
#                                       use_sde=self.use_sde,
#                                       sde_sample_freq=self.sde_sample_freq,
#                                       rollout_buffer_class=self.rollout_buffer_class,
#                                       rollout_buffer_kwargs=self.rollout_buffer_kwargs,
#                                       target_kl=self.target_kl,
#                                       policy_kwargs=self.policy_kwargs,
#                                       device=self.device,
#                                       verbose=self.verbose,
#                                       seed=self.seed)
#         return ppo_model
# class GCN_PPO(PPO):
#     def __init__(self, path, seed):
#
#         name = "GCN-PPO"
#
#         extractor = GraphExtractor
#         extractor_kwargs = dict(
#             ignored_keys=["gen_p", "gen_p_before_curtail"]
#         )
#
#
#         super().__init__(path, seed, name, extractor, extractor_kwargs)
#
#         gnn = GCN(
#             in_channels=5,
#             hidden_channels=100,
#             out_channels=15,
#             num_layers=2, dropout=0.4
#         ).to(device=self.device)
#
#         self.gnn = th.compile(gnn)
#
#     def _init_env(self, name):
#         print("Creating Graph env - " + name)
#         env = env_graph(name, self.reward, self.seed, self.gnn)
#         return env
#
#
# class GAT_PPO(PPO):
#     def __init__(self, path, seed):
#
#         name = "GCN-PPO"
#
#         extractor = GraphExtractor
#         extractor_kwargs = dict(
#             ignored_keys=["gen_p", "gen_p_before_curtail"]
#         )
#
#         super().__init__(path, seed, name, extractor, extractor_kwargs)
#
#         gnn = GAT(
#             in_channels=5,
#             hidden_channels=100,
#             out_channels=15,
#             num_layers=2,
#             dropout=0.4,
#             heads=5,
#             v2=False,
#             concat=True
#         ).to(device=self.device)
#
#         self.gnn = th.compile(gnn)
#
#     def _init_env(self, name):
#         print("Creating Graph env - " + name)
#         env = env_graph(name, self.reward, self.seed, self.gnn)
#         return env
#
# class SAGE_PPO(PPO):
#     def __init__(self, path, seed):
#
#         name = "SAGE-PPO"
#
#         extractor = GraphExtractor
#         extractor_kwargs = dict(
#             ignored_keys=["gen_p", "gen_p_before_curtail"]
#         )
#
#
#         super().__init__(path, seed, name, extractor, extractor_kwargs)
#
#         gnn = GraphSAGE(
#             in_channels=5,
#             hidden_channels=100,
#             out_channels=15,
#             num_layers=2, dropout=0.4
#         ).to(device=self.device)
#
#         self.gnn = th.compile(gnn)
#
#     def _init_env(self, name):
#         print("Creating Graph env - " + name)
#         env = env_graph(name, self.reward, self.seed, self.gnn)
#         return env
#
class PPO(BaseModel):

    def __init__(self, path, seed, name="PPO", extractor=None, extractor_kwargs=None):
        self.name = name
        self.path = path
        self.seed = seed

        super().__init__(s, seed, name)

        # Policy Parameters

        self.model = stable_baselines3.PPO
        self.verbose = 0
        self.device = get_device("auto")

        # self.gamma = 0.85  # Discount Factor
        # self.learning_rate = 1e-4  # Learning Rate
        # self.ent_coef = "auto"
        # self.use_sde = False
        # self.sde_sample_freq = -1
        #
        # self.buffer_size = int(1e6)  # Replay Buffer Size
        # self.batch_size = 256  # Number of Samples per Minibatch
        # self.tau = 0.005  # Target Smoothing Coefficient
        # self.target_update_interval = 1  # Target Update Interval
        # self.gradient_steps = 1  # Gradient Steps
        # self.learning_starts = int(5e3)
        #
        # self.n_steps = 5
        # self.gae_lambda = 1.0
        # self.vf_coef = 0.5
        # self.max_grad_norm = 0.5
        # self.use_rms_prop = True
        # self.rms_prop_eps = 1e-5
        # self.normalize_advantage = False
        # self.rollout_buffer_kwargs = None
        # self.rollout_buffer_class = None

    def _init_env(self, name, config):
        obs_attr = [
            "gen_p",
            # 'gen_q',
            # 'gen_theta',
            # 'gen_v',
            "gen_p_before_curtail",
            "load_p",
            "load_q",
            # 'load_theta',
            # 'load_v',
            "line_status",
            "rho",
            "step",
        ]

        reward = RESPenaltyReward(res_bonus=config["res_bonus"])
        print("Creating MultiInput env - " + name)
        env = env_multi(name, obs_attr, reward, self.seed)
        return env

    def _setup_train(self, env, config):
        net_arch = [config["num_units_layer"]] * config[
            "num_hidden_layers"
        ]  # Network Architecture
        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=config["activation_fn"],
            optimizer_class=config["optimizer"],
        )

        ppo_model = self.model(
            "MultiInputPolicy",
            env,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            clip_range_vf=config["clip_range_vf"],
            normalize_advantage=config["normalize_advantage"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            use_sde=config["use_sde"],
            sde_sample_freq=config["sde_sample_freq"],
            rollout_buffer_class=config["rollout_buffer_class"],
            rollout_buffer_kwargs=config["rollout_buffer_kwargs"],
            target_kl=config["target_kl"],
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=self.verbose,
            seed=self.seed,
        )
        return ppo_model


class GCN_PPO(PPO):
    def __init__(self, path, seed, name="GCN-PPO"):

        super().__init__(path, seed, name)

    def _init_env(self, name, config):
        print("Creating Graph env - " + name)

        gnn = GCN(
            in_channels=config["gnn_in_channels"],
            hidden_channels=config["gnn_hidden_channels"],
            out_channels=config["gnn_out_channels"],
            num_layers=config["gnn_num_layers"],
            dropout=config["gnn_dropout"],
        ).to(device=self.device)

        reward = RESPenaltyReward(res_penalty=config["res_bonus"])
        # reward = EpisodeDurationReward()

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

        ppo_model = self.model(
            "MultiInputPolicy",
            env,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            clip_range_vf=config["clip_range_vf"],
            normalize_advantage=config["normalize_advantage"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            use_sde=config["use_sde"],
            sde_sample_freq=config["sde_sample_freq"],
            rollout_buffer_class=config["rollout_buffer_class"],
            rollout_buffer_kwargs=config["rollout_buffer_kwargs"],
            target_kl=config["target_kl"],
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=self.verbose,
            seed=self.seed,
        )
        return ppo_model


class GAT_PPO(PPO):
    def __init__(self, path, seed, name="GAT-PPO"):

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

        reward = RESPenaltyReward(res_penalty=config["res_bonus"])

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

        ppo_model = self.model(
            "MultiInputPolicy",
            env,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            clip_range_vf=config["clip_range_vf"],
            normalize_advantage=config["normalize_advantage"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            use_sde=config["use_sde"],
            sde_sample_freq=config["sde_sample_freq"],
            rollout_buffer_class=config["rollout_buffer_class"],
            rollout_buffer_kwargs=config["rollout_buffer_kwargs"],
            target_kl=config["target_kl"],
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=self.verbose,
            seed=self.seed,
        )
        return ppo_model


class SAGE_PPO(PPO):
    def __init__(self, path, seed, name="SAGE-PPO"):

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

        ppo_model = self.model(
            "MultiInputPolicy",
            env,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            clip_range_vf=config["clip_range_vf"],
            normalize_advantage=config["normalize_advantage"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            use_sde=config["use_sde"],
            sde_sample_freq=config["sde_sample_freq"],
            rollout_buffer_class=config["rollout_buffer_class"],
            rollout_buffer_kwargs=config["rollout_buffer_kwargs"],
            target_kl=config["target_kl"],
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=self.verbose,
            seed=self.seed,
        )
        return ppo_model
