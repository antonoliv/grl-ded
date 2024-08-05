import stable_baselines3
import torch as th
import torch_geometric

from grl.environment.reward.res_penalty_reward import RESPenaltyReward

sac_params = {
    "class": stable_baselines3.SAC,
    "policy": "MultiInputPolicy",
    "learning_rate": 1e-4,
    "gamma": 0.85,
    "ent_coef": "auto",
    "gradient_steps": 1,
    "buffer_size": int(1e6),
    "batch_size": 256,
    "tau": 0.001,
    "target_update_interval": 1,
    "learning_starts": int(1e3),
    "train_freq": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "action_noise": None,
    "optimizer": th.optim.Adam,  # Optimizer
    "activation_fn": th.nn.ReLU,  # Activation Function
    "num_units_layer": 128,
    "num_hidden_layers": 6,
}

gnn_params = {
    "in_channels": 6,
    "hidden_channels": 18,
    "num_layers": 2,
    "out_channels": 6,
    "dropout": 0.1,
    "act": "relu",
    "act_first": False,
    "act_kwargs": None,
    "norm": None,
    "norm_kwargs": None,
    "jk": None,
    "aggr": "sum",
    "aggr_kwargs": None,
    "flow": "source_to_target",
    "node_dim": -2,
    "decomposed_layers": 1,

    "improved": True,
    "cached": False,
    "add_self_loops": None,
    "normalize": True,
    "bias": True,
}


env_params = {
    "env_path": "l2rpn_icaps_2021_large",
    "reward": RESPenaltyReward(0.4),
    "obs_scaled": False,
    "obs_step": True,
    "act_no_curtail": False,
    "act_limit_inf": True,
    "climit_type": "sqrt",
    "climit_end": 7200,
    "climit_low": 0.4,
    "climit_factor": 3,
}


train_ep = 10000
eval_ep = 1000

seed = 123433334
from grl.model import SAC, GCN_SAC, GAT_SAC, SAGE_SAC

m = GCN_SAC(
    seed,
    "gcn_sac/8/improved",
    1,
    sac_params.copy(),
    env_params.copy(),
    gnn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

gnn_params['act_first'] = True
gnn_params['improved'] = False

m = GCN_SAC(
    seed,
    "gcn_sac/8/act_first",
    1,
    sac_params.copy(),
    env_params.copy(),
    gnn_params.copy(),
)
m.train_and_validate(train_ep, eval_ep)

gnn_params['improved'] = True

m = GCN_SAC(
    seed,
    "gcn_sac/8/act_first_improved",
    1,
    sac_params.copy(),
    env_params.copy(),
    gnn_params.copy(),
)
m.train_and_validate(train_ep, eval_ep)
