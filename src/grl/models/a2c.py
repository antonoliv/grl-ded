
import torch as th
import stable_baselines3
from environment.create_env import env_graph, env_multi
from gnn.graph_extractor import GCNExtractor
from .base_model import BaseModel
from gnn.graph_extractor import FilterExtractor
from stable_baselines3.common.torch_layers import CombinedExtractor
from torch_geometric.nn.models.basic_gnn import GCN, GAT, GraphSAGE
from stable_baselines3.common.utils import get_device


class A2C(BaseModel):

    def __init__(self, path, seed, name="A2C", extractor=None, extractor_kwargs=None):

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
            features_extractor_kwargs=extractor_kwargs,
        )

        self.model = stable_baselines3.A2C
        self.verbose = 0
        self.device = get_device("auto")

        self.gamma = 0.99  # Discount Factor
        self.learning_rate = 7e-4  # Learning Rate
        self.ent_coef = "auto"
        self.use_sde = False
        self.sde_sample_freq = -1
        self.ent_coef = 0.0

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
        a2c_model = self.model("MultiInputPolicy",
                                          env,
                                          learning_rate=self.learning_rate,
                                          n_steps=self.n_steps,
                                          verbose=self.verbose,
                                          gamma=self.gamma,
                                          gae_lambda=self.gae_lambda,
                                          policy_kwargs=self.policy_kwargs,
                                          ent_coef=self.ent_coef,
                                          vf_coef=self.vf_coef,
                                          max_grad_norm=self.max_grad_norm,
                                          use_sde=self.use_sde,
                                          sde_sample_freq=self.sde_sample_freq,
                                          use_rms_prop=self.use_rms_prop,
                                          rms_prop_eps=self.rms_prop_eps,
                                          normalize_advantage=self.normalize_advantage,
                                          rollout_buffer_kwargs=self.rollout_buffer_kwargs,
                                          rollout_buffer_class=self.rollout_buffer_class,
                                          device=self.device,
                                          seed=self.seed)
        return a2c_model
class GCN_A2C(A2C):
    def __init__(self, path, seed):

        name = "GCN-A2C"

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


class GAT_A2C(A2C):
    def __init__(self, path, seed):

        name = "GCN-A2C"

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

class SAGE_A2C(A2C):
    def __init__(self, path, seed):

        name = "SAGE-A2C"

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
