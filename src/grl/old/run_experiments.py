import time

from models.sac import SAC, GCN_SAC


def seed(path):
    # SEED = 123456789
    SEED = int(time.time())

    # Save the seed
    with open(path + "seed.txt", "w") as file:
        file.write(f"Seed: {SEED}\n")

    return SEED


def train_model(name, env_name, model_class, base_path, seed, train_ep, eval_ep):
    path = base_path + name + "/"
    model = model_class(path, seed)
    model.train_and_validate(env_name, train_ep, eval_ep)


# Path to save the experiments
path = "/home/treeman/school/dissertation/src/grl/experiments/"

# Environment Name
env_name_14 = "l2rpn_case14_sandbox"
env_name_36 = "l2rpn_icaps_2021_small"
env_name_118 = "l2rpn_idf_2023"


train_ep = 5000  # Number of Episodes
eval_ep = 250  # Number of Evaluations

test_path = path + "grl4/"


SEED = 123213123

models = [
    ("sac", SAC, env_name_14),
    ("gcn_sac", GCN_SAC, env_name_14),
    ("sac", SAC, env_name_36),
    ("gcn_sac", GCN_SAC, env_name_36),
    ("sac", SAC, env_name_118),
    ("gcn_sac", GCN_SAC, env_name_118),
]

i = 0
for model in models:
    path = test_path + model[0] + "/"
    m = model[1](path, SEED)
    m.train_and_validate(model[2], train_ep, eval_ep)
