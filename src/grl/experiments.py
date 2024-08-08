import stable_baselines3
import torch as th
import torch_geometric
import json
from grid2op.Reward import EconomicReward

from grl.environment.reward import RESBonusReward
from grl.environment.reward.res_penalty_reward import RESPenaltyReward

def save_params(env_params, model_params, gnn_params):
    """
    Save the parameters to a file.

    :param params:  parameters
    """

    params = {
        "env": env_params.copy(),
        "model": model_params.copy(),
        "gnn": gnn_params.copy(),
    }
    # Save the parameters
    with open('params.json', 'r') as file:
        json.dump(params, file)

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
    "num_layers": 6,
    "out_channels": 6,
    "dropout": 0.1,
    "act": "relu",
    "act_first": True,
    "act_kwargs": None,
    "norm": None,
    "norm_kwargs": None,
    "jk": None,
    "aggr": "max",
    "aggr_kwargs": None,
    "flow": "source_to_target",
    "decomposed_layers": 1,
}

gcn_params = {
    "improved": False,
    "cached": False,
    "add_self_loops": None,
    "normalize": True,
    "node_dim": -2,
    "bias": True,
}

gat_params = {
    "heads": 3,
    "v2": True,
    "concat": True,
    "negative_slope": 0.2,
    "add_self_loops": True,
    "edge_dim": None,
    "fill_value": "mean",
    "bias": True,
}

sage_params = {
    "aggr": "mean",
    "normalize": False,
    "root_weight": True,
    "node_dim": -2,
    "project": False,
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
    "climit_end": 1,
    "climit_low": 0.4,
    "climit_factor": 3,
}

gcn_params.update(gnn_params)
gat_params.update(gnn_params)
sage_params.update(gnn_params)

train_ep = 1
eval_ep = 1

seed = 123433334
from grl.model import SAC, GCN_SAC, GAT_SAC, SAGE_SAC

####################################################################################################
# BASE MODELS
####################################################################################################
m = SAC(
    seed,
    "final/sac/base",
    1,
    sac_params.copy(),
    env_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

m = GCN_SAC(
    seed,
    "final/gcn_sac/base",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

####################################################################################################
# NO CURTAILMENT
####################################################################################################

env_params['act_no_curtail'] = True

m = GCN_SAC(
    seed,
    "final/gcn_sac/no_curtail",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

env_params['act_no_curtail'] = False

####################################################################################################
# NO LIMIT
####################################################################################################

env_params['act_limit_inf'] = False

m = GCN_SAC(
    seed,
    "final/gcn_sac/no_limit",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

env_params['act_limit_inf'] = True

####################################################################################################
# CURTAIL BOUND SMOOTHING
####################################################################################################

# NO SMOOTHING
env_params['climit_type'] = None

m = GCN_SAC(
    seed,
    "final/gcn_sac/no_smoothing",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

env_params['climit_type'] = "fixed"

m = GCN_SAC(
    seed,
    "final/gcn_sac/fixed_smoothing",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

env_params['climit_type'] = "linear"

m = GCN_SAC(
    seed,
    "final/gcn_sac/linear_smoothing",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

env_params['climit_type'] = "sqrt"

####################################################################################################
# REWARD
####################################################################################################

# ECONOMIC
env_params['reward'] = EconomicReward()

m = GCN_SAC(
    seed,
    "final/gcn_sac/economic",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

# PENALTY 0.2
env_params['reward'] = RESPenaltyReward(0.2)

m = GCN_SAC(
    seed,
    "final/gcn_sac/penalty_2",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

# PENALTY 0.6
env_params['reward'] = RESPenaltyReward(0.6)

m = GCN_SAC(
    seed,
    "final/gcn_sac/penalty_6",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

# BONUS 0.2
env_params['reward'] = RESBonusReward(0.2)

m = GCN_SAC(
    seed,
    "final/gcn_sac/bonus_2",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

# BONUS 0.4
env_params['reward'] = RESBonusReward(0.4)

m = GCN_SAC(
    seed,
    "final/gcn_sac/bonus_4",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

# BONUS 0.6
env_params['reward'] = RESBonusReward(0.6)

m = GCN_SAC(
    seed,
    "final/gcn_sac/bonus_6",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

env_params['reward'] = RESPenaltyReward(0.4)


####################################################################################################
# GNN COMPARISON
####################################################################################################

# GAT
m = GAT_SAC(
    seed,
    "final/gat_sac/36",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

# # SAGE
# m = SAGE_SAC(
#     seed,
#     "final/sage_sac/36",
#     1,
#     sac_params.copy(),
#     env_params.copy(),
#     sage_params.copy(),
# )

# m.train_and_validate(train_ep, eval_ep)

####################################################################################################
# 118 Scenario
####################################################################################################

env_params["env_path"] = "l2rpn_idf_2023"
eval_ep = 83

m = SAC(
    seed,
    "final/sac/118",
    1,
    sac_params.copy(),
    env_params.copy()
)

m.train_and_validate(train_ep, eval_ep)


env_params["env_path"] = "l2rpn_idf_2023"

m = GCN_SAC(
    seed,
    "final/gcn_sac/118",
    1,
    sac_params.copy(),
    env_params.copy(),
    gcn_params.copy()
)

m.train_and_validate(train_ep, eval_ep)

# GAT
m = GAT_SAC(
    seed,
    "final/gat_sac/118",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

# # SAGE
# m = SAGE_SAC(
#     seed,
#     "final/sage_sac/118",
#     1,
#     sac_params.copy(),
#     env_params.copy(),
#     sage_params.copy(),
# )

# m.train_and_validate(train_ep, eval_ep)
