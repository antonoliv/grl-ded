import grid2op
import numpy as np
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
    from gymnasium.wrappers import RescaleAction

    env.action_space = BoxGymnasiumActSpace(env.get_wrapper_attr('init_env').action_space, attr_to_keep=['curtail', 'redispatch'])  # Convert Action Space to Box

    env = RescaleAction(env, min_action=-1, max_action=1)



    obs_space = env.observation_space
    obs_space = obs_space.keep_only_attr(obs_attr)
    env.observation_space = obs_space

    # Remove the ignored attributes from the observation space

    env.action_space.seed(seed)  # for reproducible experiments
    grid_env.seed(seed)  # for reproducible experiments
    grid_env.chronics_handler.set_max_iter(2016)
    grid_env.chronics_handler.reset()

    return env


def env_graph(name, obs_attr, reward, seed):
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
    from gymnasium.wrappers import RescaleAction


    conv_rmin = []
    conv_diff = []
    for gen in range(grid_env.n_gen):
        if not grid_env.gen_renewable[gen]:
            conv_rmin.append(grid_env.gen_max_ramp_down[gen])
            conv_diff.append(grid_env.gen_max_ramp_up[gen] + grid_env.gen_max_ramp_down[gen])
        # else:
        #     conv_rmin.append(0)
        #     conv_diff.append(1)

    conv_rmin = np.array(conv_rmin)
    conv_diff = np.array(conv_diff)



    # from grid2op.gym_compat import ScalerAttrConverter
    # env.action_space = env.action_space.reencode_space("redispatch",
    #                                                    ScalerAttrConverter(substract=conv_rmin,
    #                                                                        divide=conv_diff))
      # Convert Action Space to Box



    from observation_space import GraphObservationSpace

    obs_space = GraphObservationSpace(grid_env.observation_space)
    env.observation_space = obs_space
    # Remove the ignored attributes from the observation space

    env.action_space = BoxGymnasiumActSpace(grid_env.action_space, attr_to_keep=["redispatch", "curtail"])
    # env = RescaleAction(env, min_action=-1, max_action=1)




    env.action_space.seed(seed)  # for reproducible experiments
    grid_env.seed(seed)  # for reproducible experiments
    grid_env.chronics_handler.set_max_iter(2016)
    grid_env.chronics_handler.reset()

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

    env.action_space = BoxGymnasiumActSpace(env.init_env.action_space, attr_to_keep=["curtail", "redispatch"])  # Convert Action Space to Box

    env.action_space.seed(seed)  # for reproducible experiments
    grid_env.seed(seed)  # for reproducible experiments
    grid_env.chronics_handler.set_max_iter(2016)
    grid_env.chronics_handler.reset()

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
