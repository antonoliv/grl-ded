from lightsim2grid import LightSimBackend

import grid2op
from grid2op.Action import DontAct
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.Reward import EconomicReward
from grid2op.gym_compat import GymnasiumEnv, BoxGymnasiumActSpace, BoxGymnasiumObsSpace
from gymnasium.wrappers import RescaleAction
from environment.observation_space import GraphObservationSpace
import numpy as np


def split_dataset(env_name: str, seed: int):
    SEED = seed
    grid2op.change_local_dir("/data_grid2op/")

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


def grid_env(name, reward, seed):
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

    grid_env.seed(seed)  # for reproducible experiments
    grid_env.chronics_handler.seed(seed)  # for reproducible experiments
    # grid_env.set_chunk_size(2016)
    grid_env.deactivate_forecast()
    grid_env.set_max_iter(2016)

    env = GymnasiumEnv(grid_env, render_mode="rgb_array")
    return env

def default_obs_space(env, obs_attr):
    obs_space = env.observation_space.keep_only_attr(obs_attr)
    env.observation_space = obs_space

def graph_obs_space(env, gnn):
    obs_space = GraphObservationSpace(env.get_wrapper_attr('init_env').observation_space, gnn)
    env.observation_space = obs_space

def act_space(env, seed):
    grid_env = env.get_wrapper_attr('init_env')
    r_high = []
    r_low = []
    c_high = []
    c_low = []
    for gen in range(grid_env.n_gen):
        if grid_env.gen_redispatchable[gen]:
            r_high.append(grid_env.gen_max_ramp_up[gen])
            r_low.append(- grid_env.gen_max_ramp_down[gen])
        if grid_env.gen_renewable[gen]:
            c_high.append(grid_env.gen_max_ramp_up[gen])
            c_low.append(- grid_env.gen_max_ramp_down[gen])
    r_high = np.array(r_high)
    r_low = np.array(r_low)
    c_high = np.array(c_high)
    c_low = np.array(c_low)
    env.action_space = BoxGymnasiumActSpace(grid_env.action_space,
                                            attr_to_keep=['redispatch', 'curtail'],
                                            multiply={
                                                "redispatch": (r_high - r_low) / 2,
                                                "curtail": 1 / 2
                                            },
                                            add={
                                                "redispatch": (r_high - r_low) / 2 + r_low,
                                                "curtail": 1/2
                                            })  # Convert Action Space to Box
    env.action_space.seed(seed)  # for reproducible experiments

def env_multi(name, obs_attr, reward, seed):
    # Backend for the environment

    env = grid_env(name, reward, seed)
    act_space(env, seed)

    

    default_obs_space(env, obs_attr)
    return env


def env_graph(name, reward, seed, gnn):

    env = grid_env(name, reward, seed)

    act_space(env, seed)

    graph_obs_space(env, gnn)
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

    # for reproducible experiments
    grid_env.seed(seed)  # for reproducible experiments
    grid_env.chronics_handler.set_max_iter(2016)
    grid_env.chronics_handler.set_chunk_size(2016)
    grid_env.chronics_handler.reset()

    # Create the gym environment
    env = GymnasiumEnv(grid_env, render_mode="rgb_array")

    env.observation_space = BoxGymnasiumObsSpace(env.init_env.observation_space, attr_to_keep=obs_attr)

    env.action_space = BoxGymnasiumActSpace(env.init_env.action_space,
                                            attr_to_keep=["curtail", "redispatch"])  # Convert Action Space to Box

    env.action_space.seed(seed)

    # Remove the ignored attributes from the observation space

    return env


def env_test_val(train_name, val_name, obs_attr, reward, seed, graph: bool):
    if graph:
        train_env = env_graph(train_name, obs_attr, reward, seed)
        val_env = env_graph(val_name, obs_attr, reward, seed)
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
