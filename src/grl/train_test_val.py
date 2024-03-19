import grid2op

from create_env import create_environment

SEED = 1234
env_name = "l2rpn_case14_sandbox"  # or any other...
env = create_environment(env_name, SEED)[1]

# extract 1% of the "chronics" to be used in the validation environment. The other 99% will
# be used for test
nm_env_train, nm_env_val, nm_env_test = env.train_val_split_random(pct_val=1, add_for_test="test", pct_test=1)

# and now you can use the training set only to train your agent:
print(f"The name of the training environment is \"{nm_env_train}\"")
print(f"The name of the validation environment is \"{nm_env_val}\"")
print(f"The name of the test environment is \"{nm_env_test}\"")
env_train = grid2op.make(nm_env_train)
