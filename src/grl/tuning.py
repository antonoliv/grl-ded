import time

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


def seed(path):
    # SEED = 123456789
    SEED = int(time.time())

    # Save the seed
    with open(path + "seed.txt", "w") as file:
        file.write(f"Seed: {SEED}\n")

    return SEED


# Path to save the experiments
path = "/home/treeman/school/dissertation/src/grl/experiments/"

# Environment Name
# env_path = "l2rpn_case14_sandbox"
env_name = "l2rpn_icaps_2021_small"
# env_path = "l2rpn_idf_2023"


test_path = path + "test/"

path = "gcn_sac/"

SEED = 123213123

import torch as th

config = {
    "path": test_path + path,  #
    "seed": SEED,  #
    "env_path": env_name,  #
    "res_bonus": 0.4,  #
    "train_episodes": 500,  #
    "eval_episodes": 150,  #
    "learning_rate": tune.choice([1e-5, 1e-4, 1e-3]),  #
    "gamma": tune.uniform(0.80, 0.99),  #
    "use_sde": False,  #
    "ent_coef": "auto",  #
    "optimizer": tune.choice([th.optim.Adam, th.optim.AdamW]),  # Optimizer
    "activation_fn": th.nn.ReLU,  # Activation Function
    "num_units_layer": tune.choice([48, 128, 256, 512]),  #
    "num_hidden_layers": tune.choice([2, 4, 6]),  #
    "sde_sample_freq": -1,  #
    "gradient_steps": tune.choice([1, 5, 10]),  #
    "buffer_size": tune.choice([int(1e5), int(5e5), int(1e6)]),  #
    "batch_size": tune.choice([64, 128, 256, 512]),  #
    "tau": tune.loguniform(0.001, 0.01),  #
    "target_update_interval": tune.choice([1, 10, 100]),  #
    "learning_starts": tune.choice([int(1e3), int(5e3), int(5e4)]),  #
    "train_freq": tune.choice([1, 10, 100]),  #
    "gnn_in_channels": 6,  #
    "gnn_hidden_channels": tune.choice([18, 36, 72]),  #
    "gnn_out_channels": tune.choice([6, 18, 36]),  #
    "gnn_num_layers": tune.choice([2, 3, 4]),  #
    "gnn_dropout": tune.choice([0.0, 0.2, 0.4]),  #
    # action_noise=None,
    # replay_buffer_class=None,
    # replay_buffer_kwargs=None,
    # optimize_memory_usage=False,
    # target_entropy="auto",
    # use_sde_at_warmup=False,
}


ray.init()

# Define the scheduler
scheduler = ASHAScheduler(
    metric="mean_reward", mode="max", max_t=10, grace_period=5, reduction_factor=2
)


# gcn_sac = GCN_SAC(test_path + path, SEED)
# gcn_sac.train_and_validate(config)
# tune.with_resources
#
# # Run the experiment
# analysis = tune.run(
#     gcn_sac.train_and_validate,
#     config=config,
#     num_samples=5,
#     scheduler=scheduler,
#     resources_per_trial={"cpu": 1, "gpu": 0.2},
# )

from models.experiment import Experiment
from ray import train, tune

tuner = tune.Tuner(
    tune.with_resources(Experiment, resources={"cpu": 1, "gpu": 1}),
    param_space=config,
    run_config=train.RunConfig(stop={"training_iterations": 1}),
    tune_config=tune.TuneConfig(
        num_samples=20, max_concurrent_trials=4, scheduler=scheduler
    ),
)

results = tuner.fit()

print(results.get_dataframe())
results.get_dataframe().to_csv(test_path + "parameters.csv", index=False)
