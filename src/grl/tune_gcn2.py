import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from grl.model import GCN_SAC

import stable_baselines3
import torch as th
import torch_geometric

from grl.environment.reward.res_penalty_reward import RESPenaltyReward

import settings

EXPERIMENTS = settings.EXPERIMENTS

sac_params = {
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
    "hidden_channels": tune.grid_search([6, 18]),
    "num_layers": tune.grid_search([1, 2]),
    "out_channels": 6,
    "dropout": 0.1,
    "act": "relu",
    "act_first": True, 
    "act_kwargs": None,
    "norm": None,
    "norm_kwargs": None,
    "jk": None, # Check
    "aggr": tune.grid_search(["sum", "mean", "min", "max", "mul"]),
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
    "act_no_curtail": False,
    "act_limit_inf": True,
    "climit_type": "sqrt",
    "climit_end": 2400,
    "climit_low": 0.4,
    "climit_factor": 3,
}

params = {
    "name": "gcn_sac/gnn_tune2",
    "class": GCN_SAC,
    "seed": 123433334,
    "verbose": 1,
    "train_episodes": 3000,
    "eval_episodes": 300,
    "sac_params": sac_params,
    "gnn_params": gnn_params,
    "env_params": env_params,
}




ray.init()

# Define the scheduler
scheduler = ASHAScheduler(
    metric="mean_reward", mode="max", max_t=10, grace_period=5, reduction_factor=2
)

from grl.experiment import Experiment
from ray import train, tune

tuner = tune.Tuner(
    tune.with_resources(Experiment, resources={"cpu": 1, "gpu": 1}),
    param_space=params,
    run_config=train.RunConfig(stop={"training_iteration": 1}),
    tune_config=tune.TuneConfig(
        num_samples=1, max_concurrent_trials=5, scheduler=scheduler
    ),
)

results = tuner.fit()

print(results.get_dataframe())

results.get_dataframe().to_csv(settings.EXPERIMENTS + params['name'] + "/parameters.csv", index=False)
