import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import grid2op
from grid2op.Agent import RandomAgent
from lightsim2grid import LightSimBackend
from grid2op.gym_compat import GymnasiumEnv, BoxGymnasiumActSpace
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from grid2op.Reward import EconomicReward
from gymnasium.wrappers import FlattenObservation
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.Action import DontAct
from torch import nn

path = "./models/sac/"

total_timesteps=int(1e4)

optimizer="Adam"
gamma=0.85                  # Discount Factor
learning_rate=1e-4          # Learning Rate
buffer_size=1000000         # Replay Buffer Size
activation_fn=nn.ReLU
num_hidden_layers=6
num_units_layer=256
batch_size=256              # Number of Samples per Minibatch
entropy_threshold=1
cost_threshold=1
nonlinearity="ReLU"
tau=0.005                   # Target Smoothing Coefficient
target_update_interval=1    # Target Update Interval
gradient_steps=1            # Gradient Steps

backend_class = LightSimBackend # Backend for the environment

# Create the train environment
nm_env_train = "/home/treeman/school/dissertation/src/grl/data_grid2op/l2rpn_case14_sandbox_test/"
train_grid_env = grid2op.make(nm_env_train, 
                                backend=backend_class(), 
                                reward_class=EconomicReward, 
                                opponent_attack_cooldown=999999,
                                opponent_attack_duration=0,
                                opponent_budget_per_ts=0,
                                opponent_init_budget=0,
                                opponent_action_class=DontAct,
                                opponent_class=BaseOpponent,
                                opponent_budget_class=NeverAttackBudget)
train_env = GymnasiumEnv(train_grid_env, render_mode="rgb_array")
train_env.action_space = BoxGymnasiumActSpace(train_grid_env.action_space)

net_arch=[num_units_layer] * num_hidden_layers
policy_kwargs= dict(
    net_arch=net_arch,
    activation_fn=activation_fn
)

# Instantiate the agent
model = SAC("MultiInputPolicy",
            train_env,
            verbose=1,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            target_update_interval=target_update_interval,
            gradient_steps=gradient_steps,
            policy_kwargs=policy_kwargs)

from data_collect import CollectCallback
eval_callback = CollectCallback(save_path=path + "data/train/")

# Train the agent and display a progress bar
model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
# Save the agent
model.save(path + "model")
del model  # delete trained model to demonstrate loading





# Create the validation environment
nm_env_val = "/home/treeman/school/dissertation/src/grl/data_grid2op/l2rpn_case14_sandbox_test/"
val_grid_env = grid2op.make(nm_env_val,
                            backend=backend_class(),
                            reward_class=EconomicReward,
                            opponent_attack_cooldown=999999,
                            opponent_attack_duration=0,
                            opponent_budget_per_ts=0,
                            opponent_init_budget=0,
                            opponent_action_class=DontAct,
                            opponent_class=BaseOpponent,
                            opponent_budget_class=NeverAttackBudget)
val_env = GymnasiumEnv(val_grid_env, render_mode="rgb_array")
val_env.action_space = BoxGymnasiumActSpace(val_grid_env.action_space)

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = SAC.load(path + "model", env=val_env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print("Mean Reward: " + str(mean_reward))
print("Std Reward: " + str(std_reward))

