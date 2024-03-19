import grid2op
from grid2op.Action import DontAct
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.Reward import EconomicReward
from grid2op.gym_compat import GymnasiumEnv, BoxGymnasiumActSpace
from lightsim2grid import LightSimBackend

from ignored_obs import ignored


def create_environment(name, seed):
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

    grid_env.chronics_handler.set_chunk_size(100)  # Chunk Size
    grid_env.seed(seed)  # for reproducible experiments

    # Create the gym environment
    env = GymnasiumEnv(grid_env, render_mode="rgb_array")

    env.action_space = BoxGymnasiumActSpace(grid_env.action_space)  # Convert Action Space to Box

    # Remove the ignored attributes from the observation space
    obs_space = env.observation_space
    obs_space = obs_space.ignore_attr(ignored)
    env.observation_space = obs_space

    env.action_space.seed(seed)  # for reproducible experiments
    env.render_mode = "rgb_array"  # Force render mode

    return [env, grid_env]
