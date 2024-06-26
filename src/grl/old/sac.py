import time

from models.sac import SAC


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
# env_path = "l2rpn_icaps_2021_small"
# env_path = "l2rpn_idf_2023"

train_ep = 5000  # Number of Episodes
eval_ep = 250  # Number of Evaluations

test_path = path + "test/"


path = "36sac/"

SEED = 123213123


sac = SAC(test_path + path, SEED)
sac.train_and_validate(env_name, train_ep, eval_ep)
