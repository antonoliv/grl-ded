import torch as th

from environment.utils import env_test_val
from environment.observation_space import get_obs_keys
from environment.reward.res_penalty_reward import RESPenaltyReward
import os
import time
from stable_baselines3.common.torch_layers import CombinedExtractor
import stable_baselines3
from torch import nn

optimizer = th.optim.Adam  # Optimizer TODO
gamma = 0.85  # Discount Factor
learning_rate = 1e-4  # Learning Rate
buffer_size = int(1e6)  # Replay Buffer Size
activation_fn = nn.ReLU  # Activation Function
num_hidden_layers = 6  # Number of Hidden Layers
num_units_layer = 256  # Number of Units per Hidden Layer
batch_size = 256  # Number of Samples per Minibatch
tau = 0.005  # Target Smoothing Coefficient
target_update_interval = 1  # Target Update Interval
gradient_steps = 1  # Gradient Steps
learning_starts = int(5e3)
device = "auto"

path = "./experiments/"

# Environment Name
env_name = "l2rpn_case14_sandbox"
# env_name = "l2rpn_idf_2023"

# Environment paths
train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"
seed = 123456789
train_ep = 1000  # Number of Episodes
eval_ep = 100  # Number of Evaluations

test_path = path + "test/"

# Paths for the experiments
sac_path = "sac/"
gcn_sac_path = "gcn_sac/"

# Split the dataset
# split_dataset(env_name, SEED)

# Get the observation attributes
default_attr = get_obs_keys(False, False, False, False, False)

reward = RESPenaltyReward(res_penalty=0.4)

SEED = 234523455
# SEED = seed(test_path)


train_env, val_env = env_test_val(train_name, val_name, default_attr, reward, SEED, False)

grid_env = val_env.get_wrapper_attr('init_env')

extractor = CombinedExtractor
extractor_kwargs = None

    # Policy Parameters
optimizer = th.optim.Adam  # Optimizer
activation_fn = th.nn.ReLU  # Activation Function
num_units_layer = 256
num_hidden_layers = 2

net_arch = [num_units_layer] * num_hidden_layers  # Network Architecture
policy_kwargs = dict(
    net_arch=net_arch,
    activation_fn=activation_fn,
    optimizer_class=optimizer,
    features_extractor_class=extractor,
    features_extractor_kwargs=extractor_kwargs
)
model = stable_baselines3.SAC("MultiInputPolicy",
                                      train_env,
                                      learning_rate=learning_rate,
                                      gradient_steps=gradient_steps,
                                      verbose=0,
                                      buffer_size=buffer_size,
                                      batch_size=batch_size,
                                      tau=tau,
                                      gamma=gamma,
                                      target_update_interval=target_update_interval,
                                      policy_kwargs=policy_kwargs,
                                      learning_starts=learning_starts,
                                      train_freq=1,
                                      action_noise=None,
                                      replay_buffer_class=None,
                                      replay_buffer_kwargs=None,
                                      optimize_memory_usage=False,
                                      ent_coef="auto",
                                      target_entropy="auto",
                                      use_sde=False,
                                      sde_sample_freq=-1,
                                      use_sde_at_warmup=False,
                                      device=device,
                                      seed=seed)

# model = stable_baselines3.SAC.load("/home/treeman/school/dissertation/src/grl/experiments/grl5/sac/model", env=train_env)

path += "data/val/"
os.makedirs(os.path.dirname(path), exist_ok=True)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.

env = model.get_env()

# Episode Metrics
cost = 0
res_waste = 0
avg_cost = []
avg_res_waste = []

n_res = 0
for i in range(0, grid_env.n_gen):
    if grid_env.gen_renewable[i] == 1:
        n_res += 1

start_time = time.time()
total_steps = 0
acc_reward = 0
length = 0
acc_rewards = []
lengths = []
total_episode = len(grid_env.chronics_handler.subpaths)

# Initialize variables
# agent = RandomAgent(env.action_space)
episode_count = 100  # i want to make lots of episode
update_interval = 0.1  # Update interval

# and now the loop starts
for i in range(episode_count):

    obs = env.reset()

    # now play the episode as usual
    while True:
        # fig = env.render()  # render the environment
        # plt.pause(update_interval)  # Show the plot after each iteration
        action, _states = model.predict(obs, deterministic=1)
        obs, reward, terminated, info = env.step(action)
        print(obs['current_step'])
        print(info[0]['exception'])

        if terminated:
            # in this case the episode is over

            break

env.close()


# Dict(
# '_shunt_bus': Box(-2147483648, 2147483647, (1,), int32),
# '_shunt_p': Box(-inf, inf, (1,), float32),
# '_shunt_q': Box(-inf, inf, (1,), float32),
# '_shunt_v': Box(-inf, inf, (1,), float32),
# 'a_ex': Box(0.0, inf, (20,), float32),
# 'a_or': Box(0.0, inf, (20,), float32),
# 'actual_dispatch': Box([-140. -120.  -70.  -70.  -40. -100.], [140. 120.  70.  70.  40. 100.], (6,), float32),
# 'attention_budget': Box(0.0, inf, (1,), float32),

# 'current_step': Box(-2147483648, 2147483647, (1,), int32),
# 'curtailment': Box(0.0, 1.0, (6,), float32),
# 'curtailment_limit': Box(0.0, 1.0, (6,), float32),
# 'curtailment_limit_effective': Box(0.0, 1.0, (6,), float32),
# 'day': Discrete(32),
# 'day_of_week': Discrete(8),
# 'delta_time': Box(0.0, inf, (1,), float32),
# 'duration_next_maintenance': Box(-1, 2147483647, (20,), int32),
# 'gen_margin_down': Box(0.0, [ 5. 10.  0.  0.  0. 15.], (6,), float32),
# 'gen_margin_up': Box(0.0, [ 5. 10.  0.  0.  0. 15.], (6,), float32),
# 'gen_p': Box(-162.01, [302.01    282.01    232.01001 232.01001 202.01    262.01   ], (6,), float32),
# 'gen_p_before_curtail': Box(-162.01, [302.01    282.01    232.01001 232.01001 202.01    262.01   ], (6,), float32),
# 'gen_q': Box(-inf, inf, (6,), float32),
# 'gen_theta': Box(-180.0, 180.0, (6,), float32),
# 'gen_v': Box(0.0, inf, (6,), float32),
# 'hour_of_day': Discrete(24),
# 'is_alarm_illegal': Discrete(2),
# 'line_status': MultiBinary(20),
# 'load_p': Box(-inf, inf, (11,), float32),
# 'load_q': Box(-inf, inf, (11,), float32),
# 'load_theta': Box(-180.0, 180.0, (11,), float32),
# 'load_v': Box(0.0, inf, (11,), float32),
# 'max_step': Box(-2147483648, 2147483647, (1,), int32),
# 'minute_of_hour': Discrete(60),
# 'month': Discrete(13),
# 'p_ex': Box(-inf, inf, (20,), float32),
# 'p_or': Box(-inf, inf, (20,), float32),
# 'q_ex': Box(-inf, inf, (20,), float32),
# 'q_or': Box(-inf, inf, (20,), float32),
# 'rho': Box(0.0, inf, (20,), float32),
# 'target_dispatch': Box([-140. -120.  -70.  -70.  -40. -100.], [140. 120.  70.  70.  40. 100.], (6,), float32),
# 'thermal_limit': Box(0.0, inf, (20,), float32),
# 'theta_ex': Box(-180.0, 180.0, (20,), float32),
# 'theta_or': Box(-180.0, 180.0, (20,), float32),
# 'time_before_cooldown_line': Box(0, 10, (20,), int32),
# 'time_before_cooldown_sub': Box(0, 0, (14,), int32),
# 'time_next_maintenance': Box(-1, 2147483647, (20,), int32),
# 'time_since_last_alarm': Box(-1, 2147483647, (1,), int32),
# 'timestep_overflow': Box(-2147483648, 2147483647, (20,), int32),
# 'topo_vect': Box(-1, 2, (57,), int32),
# 'v_ex': Box(0.0, inf, (20,), float32),
# 'v_or': Box(0.0, inf, (20,), float32),
# 'was_alarm_used_after_game_over': Discrete(2),
# 'year': Discrete(2100))
