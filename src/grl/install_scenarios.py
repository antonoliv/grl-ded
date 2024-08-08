from grl.environment.make import split_dataset
from torch_geometric.seed import seed_everything

val_pct = 10  # % of the dataset to use for validation

seed_everything(42)

envs = [
    "l2rpn_case14_sandbox",
    "l2rpn_icaps_2021_large",
    "l2rpn_idf_2023",
]

for env in envs:
    split_dataset(env, val_pct=val_pct)
