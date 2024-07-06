import grid2op
import gymnasium as gym
import numpy as np
import torch_geometric
from grid2op.Action import DontAct
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.Reward import EconomicReward
from grid2op.gym_compat import GymnasiumEnv, BoxGymnasiumActSpace
from gymnasium.wrappers import RescaleAction
from lightsim2grid import LightSimBackend

from environment.complete_action_space import CompleteActionSpace
from environment.no_curtail_action_space import NoCurtailActionSpace
from environment.observation_space import GraphObservationSpace

BACKEND_CLASS = LightSimBackend
MAX_ITER = 2016


def split_dataset(env_name: str, val_pct: int = 10):
    """
    Split the grid2op dataset into training and validation sets.

    :param env_name:     name of the grid2op environment
    :param val_pct:      percentage of the dataset to be used for validation
    """

    grid2op.change_local_dir("./data_grid2op/")

    # Create the train environment
    grid_env = grid2op.make(
        env_name,
        backend=BACKEND_CLASS(),
        reward_class=EconomicReward,
        opponent_attack_cooldown=999999,
        opponent_attack_duration=0,
        opponent_budget_per_ts=0,
        opponent_init_budget=0,
        opponent_action_class=DontAct,
        opponent_class=BaseOpponent,
        opponent_budget_class=NeverAttackBudget,
    )

    # Split the dataset into training and validation sets
    grid_env.train_val_split_random(pct_val=val_pct, add_for_val="val")


def grid_params(
    no_overflow_disconnection: bool = False,
    nb_timestep_overflow_allowed: int = 2,
    nb_timestep_reconnection: int = 10,
    nb_timestep_cooldown_line: int = 0,
    hard_overflow_threshold: float = 2.0,
    soft_overflow_threshold: float = 1.0,
    ignore_min_up_down_time: bool = True,
    limit_infeasible_curtailment_storage_action: bool = False,
) -> grid2op.Parameters:
    """
    Create grid2op environment parameters.

    :param no_overflow_disconnection:                       lines are not disconnected in case of overflow
    :param nb_timestep_overflow_allowed:                    number of timesteps lines can be in overflow before being disconnected
    :param nb_timestep_reconnection:                        number of timesteps for which a soft overflow is allowed
    :param nb_timestep_cooldown_line:                       number of timesteps a powerline disconnected for security motives will remain disconnected
    :param hard_overflow_threshold:                         overflow threshold for instantly disconnecting the affected line
    :param soft_overflow_threshold:                         soft overflow threshold for disconnecting the affected line after NB_TIMESTEP_OVERFLOW_ALLOWED timesteps
    :param ignore_min_up_down_time:                         ignore gen_min_uptime and gen_min_downtime
    :param limit_infeasible_curtailment_storage_action:     limit curtailment action to the "maximum feasible" intervals to prevent or reduced infeasible states
    :return: grid2op parameters
    """

    p = grid2op.Parameters.Parameters()

    # Lines are not disconnected in case of overflow
    p.NO_OVERFLOW_DISCONNECTION = no_overflow_disconnection

    # Number of timesteps lines can be in overflow before being disconnected
    p.NB_TIMESTEP_OVERFLOW_ALLOWED = np.int32(nb_timestep_overflow_allowed)

    # Number of timesteps for which a soft overflow is allowed
    p.NB_TIMESTEP_RECONNECTION = np.int32(nb_timestep_reconnection)

    # Number of timesteps a powerline disconnected for security motives will remain disconnected
    p.NB_TIMESTEP_COOLDOWN_LINE = np.int32(nb_timestep_cooldown_line)

    # Overflow threshold for instantly disconnecting the affected line
    p.HARD_OVERFLOW_THRESHOLD = np.float32(hard_overflow_threshold)

    # Soft Overflow threshold for disconnecting the affected line after NB_TIMESTEP_OVERFLOW_ALLOWED timesteps
    p.SOFT_OVERFLOW_THRESHOLD = np.float32(soft_overflow_threshold)

    # Ignore gen_min_uptime and gen_min_downtime
    p.IGNORE_MIN_UP_DOWN_TIME = ignore_min_up_down_time

    # Limit curtailment action to the "maximum feasible" intervals to prevent or reduced infeasible states
    p.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = (
        limit_infeasible_curtailment_storage_action
    )

    p.NB_TIMESTEP_COOLDOWN_SUB = np.int32(0)
    p.ENV_DC = False
    p.FORECAST_DC = False  # DEPRECATED use "change_forecast_parameters(new_param)" with "new_param.ENV_DC=..."
    p.MAX_SUB_CHANGED = np.int32(1)
    p.MAX_LINE_STATUS_CHANGED = np.int32(1)

    p.ALLOW_DISPATCH_GEN_SWITCH_OFF = True
    p.INIT_STORAGE_CAPACITY = np.float32(0.5)
    p.ACTIVATE_STORAGE_LOSS = True
    p.ALARM_BEST_TIME = np.float32(12)
    p.ALARM_WINDOW_SIZE = np.int32(12)
    p.ALERT_TIME_WINDOW = np.int32(12)
    p.MAX_SIMULATE_PER_STEP = np.int32(-1)
    p.MAX_SIMULATE_PER_EPISODE = np.int32(-1)

    return p


def grid_env(
    name: str,
    seed: int,
    reward: grid2op.Reward,
    params: grid2op.Parameters = grid_params(),
) -> gym.Env:
    """
    Create a grid2op environment and convert it to a gym environment.

    :param name:        name of the grid2op environment
    :param reward:      reward class
    :param seed:        seed for reproducible experiments
    :param params:      grid2op environment parameters
    :return:
    """

    if name is None or str.strip(name) == "":
        raise ValueError("Name is empty")
    if reward is None:
        raise ValueError("Reward is empty")
    if seed is None:
        raise ValueError("Seed is empty")
    if params is None:
        raise ValueError("Parameters are empty")

    # Create the environment
    grid_env = grid2op.make(
        name,
        backend=BACKEND_CLASS(),
        reward_class=reward,
        param=params,
        opponent_attack_cooldown=999999,
        opponent_attack_duration=0,
        opponent_budget_per_ts=0,
        opponent_init_budget=0,
        opponent_action_class=DontAct,
        opponent_class=BaseOpponent,
        opponent_budget_class=NeverAttackBudget,
    )

    grid_env.seed(seed)  # set the seed
    # grid_env.set_chunk_size(100)          # set the chunk size
    grid_env.deactivate_forecast()  # deactivate the forecast
    grid_env.set_max_iter(MAX_ITER)  # set the maximum number of iterations per episode

    return GymnasiumEnv(grid_env, render_mode="rgb_array", shuffle_chronics=True)


def act_space(env, seed, no_curtail=False, curtail_limit=0.0):
    grid_env = env.get_wrapper_attr("init_env")

    from environment.complete_action_space import CompleteActionSpace

    if no_curtail:
        env.action_space = BoxGymnasiumActSpace(
            grid_env.action_space, attr_to_keep=["redispatch"]
        )  # Convert Action Space to Box
    else:
        env.action_space = CompleteActionSpace(
            grid_env, True, curtail_limit=curtail_limit
        )
    env.action_space.seed(seed)  # for reproducible experiments


def env_multi(name, obs_attr, reward, seed):
    # Backend for the environment

    env = grid_env(name, reward, seed)
    act_space(env, seed)

    obs_space = env.observation_space.keep_only_attr(obs_attr)
    env.observation_space = obs_space
    return env


def env_graph(
    name: str,
    seed: int,
    params: grid2op.Parameters,
    reward: grid2op.Reward,
    gnn: torch_geometric.nn.models,
    obs_scaled: bool = True,
    step: bool = True,
    no_curtail: bool = False,
    curtail_limit: float = 0.0,
):
    env = grid_env(name, seed, reward, params=params)

    g_env = env.get_wrapper_attr("init_env")

    # act_space(env, seed, no_curtail=True)
    # env = RescaleAction(env, -1.0, 1.0)

    obs_space = GraphObservationSpace(
        env.init_env.observation_space, gnn, scaled=obs_scaled, step=step
    )

    env.observation_space = obs_space

    if no_curtail:
        act_space = BoxGymnasiumActSpace(
            env.init_env.action_space,
            attr_to_keep=["redispatch"],
        )  # Convert Action Space to Box
        env = RescaleAction(env, -1.0, 1.0)
    else:
        act_space = CompleteActionSpace(env.init_env, True, curtail_limit=curtail_limit)

    act_space.seed(seed)  # for reproducible experiments
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    return env


def make_env(
    env_path: str,
    seed: int,
    reward: grid2op.Reward,
    obs_graph: bool,
    obs_scaled: bool,
    obs_step: bool,
    act_no_curtail: bool,
    act_curtail_limit: float,
    gnn: torch_geometric.nn.models = None,
    grid_params: grid2op.Parameters = grid_params(),
):

    if obs_graph is None:
        raise ValueError("obs_graph is not initialized")
    if obs_scaled is None:
        raise ValueError("obs_scaled is not initialized")
    if obs_step is None:
        raise ValueError("obs_step is not initialized")
    if act_no_curtail is None:
        raise ValueError("act_no_curtail is not initialized")
    if act_curtail_limit is None:
        act_curtail_limit = 0.0

    # Create the environment
    env = grid_env(env_path, seed, reward, params=grid_params)

    g_env = env.get_wrapper_attr("init_env")

    obs_attr = [
        "gen_p",
        # 'gen_q',
        # 'gen_theta',
        # 'gen_v',
        "gen_p_before_curtail",
        "load_p",
        "load_q",
        # 'load_theta',
        # 'load_v',
        "line_status",
        "rho",
        "step",
    ]
    if obs_graph:
        obs_space = GraphObservationSpace(
            g_env, scaled=obs_scaled, gnn=gnn, step=obs_step
        )
    else:
        if obs_step:
            obs_attr = [
                "gen_p",
                "gen_p_before_curtail",
                "load_p",
                "load_q",
                "line_status",
                "rho",
                "current_step",
            ]
        else:
            obs_attr = [
                "gen_p",
                "gen_p_before_curtail",
                "load_p",
                "load_q",
                "line_status",
                "rho",
                "minute_of_hour",
                "hour_of_day",
                "day_of_week",
            ]

        obs_space = env.observation_space.keep_only_attr(obs_attr)

    if act_no_curtail:
        act_space = NoCurtailActionSpace(g_env, scaled=True)
        # act_space = BoxGymnasiumActSpace(
        # g_env.action_space, attr_to_keep=["redispatch"]
        # )
    else:
        act_space = CompleteActionSpace(
            g_env, scaled=True, curtail_limit=act_curtail_limit
        )

    env.observation_space = obs_space
    env.action_space = act_space

    env.action_space.seed(seed)

    from stable_baselines3.common.env_checker import check_env

    check_env(env)

    return env
