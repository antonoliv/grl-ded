import time

from models.sac import SAGE_SAC


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
env_name = "l2rpn_case14_sandbox"
# env_path = "l2rpn_idf_2023"

# Environment paths
train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"

train_ep = 5000  # Number of Episodes
eval_ep = 250  # Number of Evaluations

test_path = path + "grl3/"

path = "sage_sac/"

SEED = 232427376

sage_sac = SAGE_SAC(test_path + path, SEED)
sage_sac.train_and_validate(train_name, val_name, train_ep, eval_ep)
