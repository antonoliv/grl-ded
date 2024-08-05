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
    "class": torch_geometric.nn.models.GCN,
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
    "improved": False,
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
    "act_no_curtail": True,
    "act_limit_inf": False,
    "climit_type": None,
    "climit_end": 7200,
    "climit_low": 0.4,
    "climit_factor": 3,
}

# SAC -> Done
# Rewards -> RESPenalty, RESBonus, vary res term
#

train_ep = 10000
eval_ep = 1000

seed = 123433334
from grl.model import SAC, GCN_SAC

m = GCN_SAC(
    seed,
    "gcn_sac/6/no_curtail",
    1,
    sac_params.copy(),
    env_params.copy(),
    gnn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

env_params['act_no_curtail'] = False
env_params['climit_type'] = 'sqrt'

m = GCN_SAC(
    seed,
    "gcn_sac/6/curtail",
    1,
    sac_params.copy(),
    env_params.copy(),
    gnn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

env_params['act_limit_inf'] = True

m = GCN_SAC(
    seed,
    "gcn_sac/6/limit_curtail",
    1,
    sac_params.copy(),
    env_params.copy(),
    gnn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)
