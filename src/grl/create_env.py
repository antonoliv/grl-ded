import grid2op
from grid2op.Action import DontAct
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.Reward import EconomicReward
from grid2op.gym_compat import GymnasiumEnv, BoxGymnasiumActSpace, BoxGymnasiumObsSpace
from lightsim2grid import LightSimBackend

from grid_obs import get_obs_keys

def env_multi(name, seed):
    # Backend for the environment
    backend_class = LightSimBackend

    # Create the train environment
    grid_env = grid2op.make(name,
                            backend=backend_class(),
                            reward_class=EconomicReward,
                            opponent_attack_cooldown=999999,
                            opponent_attack_duration=0,
                            opponent_budget_per_ts=0,
                            opponent_init_budget=0,
                            opponent_action_class=DontAct,
                            opponent_class=BaseOpponent,
                            opponent_budget_class=NeverAttackBudget)
    

    grid_env.seed(seed)  # for reproducible experiments

    # Create the gym environment
    env = GymnasiumEnv(grid_env, render_mode="rgb_array")

    env.action_space.close()
    env.action_space = BoxGymnasiumActSpace(env.init_env.action_space)  # Convert Action Space to Box

    attr = get_obs_keys(False, False, False, False)
    obs_space = env.observation_space
    obs_space = obs_space.keep_only_attr(attr)
    env.observation_space = obs_space


    # Remove the ignored attributes from the observation space
    

    env.action_space.seed(seed)  # for reproducible experiments

    return [env, grid_env]


def env_box(name, seed):
    # Backend for the environment
    backend_class = LightSimBackend

    # Create the train environment
    grid_env = grid2op.make(name,
                            backend=backend_class(),
                            reward_class=EconomicReward,
                            opponent_attack_cooldown=999999,
                            opponent_attack_duration=0,
                            opponent_budget_per_ts=0,
                            opponent_init_budget=0,
                            opponent_action_class=DontAct,
                            opponent_class=BaseOpponent,
                            opponent_budget_class=NeverAttackBudget)

    grid_env.seed(seed)  # for reproducible experiments

    # Create the gym environment
    env = GymnasiumEnv(grid_env, render_mode="rgb_array")

    attr = get_obs_keys(False, False, False, False)
    env.observation_space.close()
    env.observation_space = BoxGymnasiumObsSpace(env.init_env.observation_space, attr_to_keep=attr)

    env.action_space.close()
    env.action_space = BoxGymnasiumActSpace(env.init_env.action_space)  # Convert Action Space to Box

    # Remove the ignored attributes from the observation space
    

    env.action_space.seed(seed)  # for reproducible experiments

    return [env, grid_env]
