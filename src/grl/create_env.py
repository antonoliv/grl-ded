import grid2op
from grid2op.Action import DontAct
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.Reward import EconomicReward
from grid2op.gym_compat import GymnasiumEnv, BoxGymnasiumActSpace, BoxGymnasiumObsSpace
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache


def split_dataset(env_name: str, seed: int):

    SEED = seed
    grid2op.change_local_dir("/home/treeman/school/dissertation/src/grl/data_grid2op/")
    
    # Backend for the environment
    backend_class = LightSimBackend

    # Create the train environment
    grid_env = grid2op.make(env_name,
                            backend=backend_class(),
                            reward_class=EconomicReward,
                            opponent_attack_cooldown=999999,
                            opponent_attack_duration=0,
                            opponent_budget_per_ts=0,
                            opponent_init_budget=0,
                            opponent_action_class=DontAct,
                            opponent_class=BaseOpponent,
                            opponent_budget_class=NeverAttackBudget
                            )

    # extract 1% of the "chronics" to be used in the validation environment. The other 99% will
    # be used for test
    nm_env_train, nm_env_val = grid_env.train_val_split_random(pct_val=10, add_for_val="val")

    # and now you can use the training set only to train your agent:
    print(f"The name of the training environment is \"{nm_env_train}\"")
    print(f"The name of the validation environment is \"{nm_env_val}\"")

def env_multi(name, obs_attr, reward, seed):
    # Backend for the environment
    backend_class = LightSimBackend

    # Create the train environment
    grid_env = grid2op.make(name,
                            backend=backend_class(),
                            reward_class=reward,
                            opponent_attack_cooldown=999999,
                            opponent_attack_duration=0,
                            opponent_budget_per_ts=0,
                            opponent_init_budget=0,
                            opponent_action_class=DontAct,
                            opponent_class=BaseOpponent,
                            opponent_budget_class=NeverAttackBudget
                            )
    

    # Create the gym environment
    env = GymnasiumEnv(grid_env, render_mode="rgb_array")

    env.action_space = BoxGymnasiumActSpace(env.init_env.action_space)  # Convert Action Space to Box

    

    obs_space = env.observation_space
    obs_space = obs_space.keep_only_attr(obs_attr)
    env.observation_space = obs_space


    # Remove the ignored attributes from the observation space
    

    env.action_space.seed(seed)  # for reproducible experiments
    env.init_env.seed(seed)  # for reproducible experiments
    env.init_env.chronics_handler.set_max_iter(2016)
    env.init_env.chronics_handler.reset()

    return env


def env_box(name, obs_attr, reward, seed):
    # Backend for the environment
    backend_class = LightSimBackend

    # Create the train environment
    grid_env = grid2op.make(name,
                            backend=backend_class(),
                            reward_class=reward,
                            opponent_attack_cooldown=999999,
                            opponent_attack_duration=0,
                            opponent_budget_per_ts=0,
                            opponent_init_budget=0,
                            opponent_action_class=DontAct,
                            opponent_class=BaseOpponent,
                            opponent_budget_class=NeverAttackBudget)

    

    # Create the gym environment
    env = GymnasiumEnv(grid_env, render_mode="rgb_array")
    
    env.observation_space = BoxGymnasiumObsSpace(env.init_env.observation_space, attr_to_keep=obs_attr)

    env.action_space = BoxGymnasiumActSpace(env.init_env.action_space)  # Convert Action Space to Box

    env.action_space.seed(seed)  # for reproducible experiments
    env.init_env.seed(seed)  # for reproducible experiments
    env.init_env.chronics_handler.set_max_iter(2016)
    env.init_env.chronics_handler.reset()

    # Remove the ignored attributes from the observation space

    return env

def env_test_val(train_name, val_name, obs_attr, reward, seed, box: bool):
    if box:
        train_env = env_box(train_name, obs_attr, reward, seed)
        val_env = env_box(val_name, obs_attr, reward, seed)
    else:
        train_env = env_multi(train_name, obs_attr, reward, seed)
        val_env = env_multi(val_name, obs_attr, reward, seed)
    return train_env, val_env


# env_name = "l2rpn_idf_2023"
# split_dataset(env_name, 123456789)
# split_dataset("l2rpn_case14_sandbox", 123456789)
# train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
# val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"
# default_attr = get_obs_keys(False, False, False, False, False)
# m_train_env, m_val_env = env_test_val(train_name, val_name, default_attr, 123456789, False)
# print(len(m_train_env.init_env.chronics_handler.subpaths))
# print(len(m_val_env.init_env.chronics_handler.subpaths))