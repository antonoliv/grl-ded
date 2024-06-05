
import torch as th
import stable_baselines3
from environment.create_env import env_graph, env_multi
from gnn.graph_extractor import GCNExtractor
from .base_model import BaseModel
from gnn.graph_extractor import FilterExtractor
from stable_baselines3.common.torch_layers import CombinedExtractor
from torch_geometric.nn.models.basic_gnn import GCN, GAT, GraphSAGE
from stable_baselines3.common.utils import get_device

class SAC(BaseModel):

    def __init__(self, path, seed, name="SAC", extractor=None, extractor_kwargs=None):

        self.name = name
        self.path = path
        self.seed = seed

        super().__init__(path, seed, name)

        if extractor is None:
            extractor = CombinedExtractor
            extractor_kwargs = None

        # Policy Parameters
        optimizer = th.optim.Adam  # Optimizer
        activation_fn = th.nn.ReLU  # Activation Function
        num_units_layer = 256
        num_hidden_layers = 2



        net_arch = [num_units_layer] * num_hidden_layers  # Network Architecture
        self.policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            optimizer_class=optimizer,
            features_extractor_class=extractor,
            features_extractor_kwargs=extractor_kwargs
        )


        self.model = stable_baselines3.SAC
        self.verbose = 0
        self.device = get_device("auto")

        self.gamma = 0.85  # Discount Factor
        self.learning_rate = 1e-4  # Learning Rate
        self.ent_coef = "auto"
        self.use_sde = False
        self.sde_sample_freq = -1

        self.buffer_size = int(1e6)  # Replay Buffer Size
        self.batch_size = 256  # Number of Samples per Minibatch
        self.tau = 0.005  # Target Smoothing Coefficient
        self.target_update_interval = 1  # Target Update Interval
        self.gradient_steps = 1  # Gradient Steps
        self.learning_starts = int(5e3)

        self.n_steps = 5
        self.gae_lambda = 1.0
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.use_rms_prop = True
        self.rms_prop_eps = 1e-5
        self.normalize_advantage = False
        self.rollout_buffer_kwargs = None
        self.rollout_buffer_class = None

    def _init_env(self, name):
        obs_attr = [
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
                    'rho',
                    'step'
                    ]

        print("Creating MultiInput env - " + name)
        env = env_multi(name, obs_attr, self.reward, self.seed)
        return env

    def _setup_train(self, env):
        sac_model = self.model("MultiInputPolicy",
                                          env,
                                          learning_rate=self.learning_rate,
                                          verbose=self.verbose,
                                          gamma=self.gamma,
                                          use_sde=False,
                                          ent_coef=self.ent_coef,
                                          policy_kwargs=self.policy_kwargs,
                                          sde_sample_freq=self.sde_sample_freq,
                                          device=self.device,
                                          seed=self.seed,
                                          gradient_steps=self.gradient_steps,
                                          buffer_size=self.buffer_size,
                                          batch_size=self.batch_size,
                                          tau=self.tau,
                                          target_update_interval=self.target_update_interval,
                                          learning_starts=self.learning_starts,
                                          train_freq=1,
                                          action_noise=None,
                                          replay_buffer_class=None,
                                          replay_buffer_kwargs=None,
                                          optimize_memory_usage=False,
                                          target_entropy="auto",
                                          use_sde_at_warmup=False)
        return sac_model


class GCN_SAC(SAC):
    def __init__(self, path, seed):

        name = "GCN-SAC"

        extractor = FilterExtractor
        extractor_kwargs = dict(
            ignored_keys=["gen_p", "gen_p_before_curtail"]
        )



        super().__init__(path, seed, name, extractor, extractor_kwargs)

        gnn = GCN(
            in_channels=5,
            hidden_channels=100,
            out_channels=15,
            num_layers=2, dropout=0.4
        ).to(device=self.device)

        self.gnn = th.compile(gnn)

    def _init_env(self, name):
        print("Creating Graph env - " + name)
        env = env_graph(name, self.reward, self.seed, self.gnn)
        return env


class GAT_SAC(SAC):
    def __init__(self, path, seed):

        name = "GCN-SAC"

        extractor = FilterExtractor
        extractor_kwargs = dict(
            ignored_keys=["gen_p", "gen_p_before_curtail"]
        )


        super().__init__(path, seed, name, extractor, extractor_kwargs)

        gnn = GAT(
            in_channels=5,
            hidden_channels=100,
            out_channels=15,
            num_layers=2,
            dropout=0.4,
            heads=5,
            v2=False,
            concat=True
        ).to(device=self.device)

        self.gnn = th.compile(gnn)

    def _init_env(self, name):
        print("Creating Graph env - " + name)
        env = env_graph(name, self.reward, self.seed, self.gnn)
        return env

class SAGE_SAC(SAC):
    def __init__(self, path, seed):

        name = "SAGE-SAC"

        extractor = FilterExtractor
        extractor_kwargs = dict(
            ignored_keys=["gen_p", "gen_p_before_curtail"]
        )


        super().__init__(path, seed, name, extractor, extractor_kwargs)

        gnn = GraphSAGE(
            in_channels=5,
            hidden_channels=100,
            out_channels=15,
            num_layers=2, dropout=0.4
        ).to(device=self.device)

        self.gnn = th.compile(gnn)

    def _init_env(self, name):
        print("Creating Graph env - " + name)
        env = env_graph(name, self.reward, self.seed, self.gnn)
        return env
