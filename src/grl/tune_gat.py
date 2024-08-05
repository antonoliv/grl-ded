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
    "num_layers": 1,
    "out_channels": 6,
    "dropout": 0.1,
    "act": "relu",
    "act_first": True,
    "act_kwargs": None,
    "norm": None,
    "norm_kwargs": None,
    "jk": None,
    "aggr": "sum",
    "aggr_kwargs": None,
    "flow": "source_to_target",
    "decomposed_layers": 1,
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

env_params = {
    "env_path": "l2rpn_icaps_2021_large",
    "reward": RESPenaltyReward(0.4),
    "obs_scaled": False,
    "obs_step": True,
    "act_no_curtail": False,
    "act_limit_inf": True,
    "climit_type": "sqrt",
    "climit_end": 4200,
    "climit_low": 0.4,
    "climit_factor": 3,
}

gat_params.update(gnn_params)

train_ep = 5000
eval_ep = 295

seed = 123433334
from grl.model import GAT_SAC

results = []
####################################################################################################
# GNN Aggr
####################################################################################################

gat_params['aggr'] = "sum"

m = GAT_SAC(
    seed,
    "final/gat_sac/sum",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

results.append(m.train_and_validate(train_ep, eval_ep))

gat_params['aggr'] = "min"

m = GAT_SAC(
    seed,
    "final/gat_sac/min",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

results.append(m.train_and_validate(train_ep, eval_ep))

gat_params['aggr'] = "max"

m = GAT_SAC(
    seed,
    "final/gat_sac/max",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

results.append(m.train_and_validate(train_ep, eval_ep))

gat_params['aggr'] = "mean"

m = GAT_SAC(
    seed,
    "final/gat_sac/mean",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

results.append(m.train_and_validate(train_ep, eval_ep))

gat_params['aggr'] = "mul"

m = GAT_SAC(
    seed,
    "final/gat_sac/mul",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

results.append(m.train_and_validate(train_ep, eval_ep))

max_r = 0
max_i = 0
for i in range(len(results)):
    if results[i]["mean_reward"] > max_r:
        max_r = results[i]["mean_reward"]
        max_i = i

aggr_lst = ["sum", "min", "max", "mean", "mul"]

gat_params['aggr'] = aggr_lst[max_i]
gat_params['aggr'] = aggr_lst[max_i]

results = []
####################################################################################################
# GNN Layers
####################################################################################################

gat_params['num_layers'] = 1

m = GAT_SAC(
    seed,
    "final/gat_sac/1_layer",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

results.append(m.train_and_validate(train_ep, eval_ep))

gat_params['num_layers'] = 2

m = GAT_SAC(
    seed,
    "final/gat_sac/2_layer",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

gat_params['num_layers'] = 3

m = GAT_SAC(
    seed,
    "final/gat_sac/3_layer",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

results.append(m.train_and_validate(train_ep, eval_ep))

gat_params['num_layers'] = 4

m = GAT_SAC(
    seed,
    "final/gat_sac/4_layer",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

results.append(m.train_and_validate(train_ep, eval_ep))

gat_params['num_layers'] = 5

m = GAT_SAC(
    seed,
    "final/gat_sac/5_layer",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

results.append(m.train_and_validate(train_ep, eval_ep))

gat_params['num_layers'] = 6

m = GAT_SAC(
    seed,
    "final/gat_sac/6_layer",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

results.append(m.train_and_validate(train_ep, eval_ep))

max_r = 0
max_i = 0
for i in range(len(results)):
    if results[i]["mean_reward"] > max_r:
        max_r = results[i]["mean_reward"]
        max_i = i

gat_params['num_layers'] = max_i + 1
gat_params['num_layers'] = max_i + 1


####################################################################################################
# GNN Heads
####################################################################################################

gat_params['heads'] = 1
# GAT
m = GAT_SAC(
    seed,
    "final/gat_sac/1_head",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

gat_params['heads'] = 2
# GAT
m = GAT_SAC(
    seed,
    "final/gat_sac/2_head",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

gat_params['heads'] = 3
# GAT
m = GAT_SAC(
    seed,
    "final/gat_sac/3_head",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)

gat_params['heads'] = 6
# GAT
m = GAT_SAC(
    seed,
    "final/gat_sac/6_head",
    1,
    sac_params.copy(),
    env_params.copy(),
    gat_params.copy(),
)

m.train_and_validate(train_ep, eval_ep)
