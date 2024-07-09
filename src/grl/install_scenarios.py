from grl.environment.make import split_dataset

val_pct = 10  # % of the dataset to use for validation

envs = [
    "l2rpn_case14_sandbox",
    "l2rpn_icaps_2021_large",
    "l2rpn_idf_2023",
]

for env in envs:
    split_dataset(env, val_pct=val_pct)
