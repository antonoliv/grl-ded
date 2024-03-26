import grid2op

from create_env import env_box


def split_dataset(env_name: str, seed: int):

    SEED = seed
    grid2op.change_local_dir("/home/treeman/school/dissertation/src/grl/data_grid2op/")
    env = env_box(env_name, SEED)[1]

    # extract 1% of the "chronics" to be used in the validation environment. The other 99% will
    # be used for test
    nm_env_train, nm_env_val, nm_env_test = env.train_val_split_random(pct_val=1, add_for_test="test", pct_test=1)

    # and now you can use the training set only to train your agent:
    print(f"The name of the training environment is \"{nm_env_train}\"")
    print(f"The name of the validation environment is \"{nm_env_val}\"")
    print(f"The name of the test environment is \"{nm_env_test}\"")
