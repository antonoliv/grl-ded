from create_env import env_multi, env_test_val
from observation_space import get_obs_keys
from reward.res_reward import DynamicEconomicReward
import stable_baselines3
from torch import nn
from core import train, validate

SEED = 1234
# SAC Hyperparameters
optimizer = "Adam"  # Optimizer TODO
gamma = 0.85  # Discount Factor
learning_rate = 1e-4  # Learning Rate
buffer_size = 1000000  # Replay Buffer Size
activation_fn = nn.ReLU  # Activation Function
num_hidden_layers = 6  # Number of Hidden Layers
num_units_layer = 256  # Number of Units per Hidden Layer
batch_size = 256  # Number of Samples per Minibatch
entropy_threshold = 1  # Entropy Threshold TODO
cost_threshold = 1  # Cost Threshold TODO
tau = 0.005  # Target Smoothing Coefficient
target_update_interval = 1  # Target Update Interval
gradient_steps = 1  # Gradient Steps

# Define the network architecture
net_arch = [num_units_layer] * num_hidden_layers
from gnn.graph_extractor import GCNExtractor
policy_kwargs = dict(
    net_arch=net_arch,
    activation_fn=activation_fn,
    features_extractor_class=GCNExtractor,
)


# Path to save the models
path = "/home/treeman/school/dissertation/src/grl/models/"

# Environment Name
env_name = "l2rpn_case14_sandbox"

# Environment paths
train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"


train_ep = 10                     # Number of Episodes
eval_ep = 100                    # Number of Evaluations


test_path = path + "test/"

# Paths for the modelssac_path = "b_sac/"

# Split the dataset
# split_dataset(env_name, SEED)

# Get the observation attributes
default_attr = get_obs_keys(False, False, False, False, False)
reward = DynamicEconomicReward(res_penalty=0.4)

seed = 123456789
from create_env import env_graph, env_box
# MultInput Policy + Default Observation Space
train_env = env_graph(train_name, default_attr, reward, SEED)



sac_model = stable_baselines3.SAC("MultiInputPolicy",
                                  train_env,
                                  learning_rate=learning_rate,
                                  gradient_steps=gradient_steps,
                                  verbose=1,
                                  buffer_size=buffer_size,
                                  batch_size=batch_size,
                                  tau=tau,
                                  gamma=gamma,
                                  target_update_interval=target_update_interval,
                                  policy_kwargs=policy_kwargs,
                                  learning_starts=100,
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
                                  seed=seed
                                  )


model = sac_model

from core import train

max_episodes = train_ep
avg_episode_length = 1500
total_timesteps = max_episodes * avg_episode_length

obs = train_env.reset()
from stable_baselines3.common.env_checker import check_env

# check_env(train_env)
# Create a callbacks
from callbacks.train_episode import EpisodeCallback
episode_callback = EpisodeCallback(max_episodes=max_episodes, verbose=0)  # Stops training after x episodes


# Train the agent and display a progress bar
model.learn(total_timesteps=500, progress_bar=True, callback=episode_callback)
grid_env = train_env.get_wrapper_attr('init_env')
eval_episodes = 10
path += "data/val/"

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.

env = model.get_env()

# Episode Metrics
cost = 0
res_waste = []
avg_cost = []
avg_res_waste = []

n_res = 0
for i in range(0, grid_env.n_gen):
    if grid_env.gen_renewable[i] == 1:
        n_res += 1

total_steps = 0
acc_reward = 0
length = 0
acc_rewards = []
lengths = []
total_episode = len(grid_env.chronics_handler.subpaths)

# Initialize variables
# agent = RandomAgent(env.action_space)
episode_count = eval_episodes  # i want to make lots of episode
update_interval = 0.1  # Update interval

# # and now the loop starts
# for i in range(episode_count):
#
#     ###################################
#     if i % total_episode == 0:
#         # I shuffle each time i need to
#         grid_env.chronics_handler.shuffle()
#     ###################################
#
#     obs = env.reset()
#
#     # now play the episode as usual
#     while True:
#         # fig = env.render()  # render the environment
#         # plt.pause(update_interval)  # Show the plot after each iteration
#
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, terminated, info = env.step(action)
#
#         acc_reward += reward
#         length += 1
#         total_steps += 1
#
#         res_waste = 0
#
#         if terminated:
#             # in this case the episode is over
#             acc_rewards.append(acc_reward)
#             avg_cost.append(cost * 288 / length)
#             avg_res_waste.append(res_waste * 288 / length)
#             lengths.append(length)
#             acc_reward = 0
#             length = 0
#             cost = 0
#             res_waste = 0
#             break

# array([[ 8.2246674e+01, -4.2346096e+01,  0.0000000e+00,  0.0000000e+00,  -3.9900578e+01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 4.1986198e+01,  5.9500000e+01, -2.3991766e+01, -4.1828262e+01,  -3.5666172e+01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  2.3725140e+01, -6.5000000e+00, -1.7225140e+01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  4.0866714e+01,  1.6992859e+01, -4.4299999e+01,  2.7542929e+01,  0.0000000e+00, -2.6171078e+01,  0.0000000e+00,  -1.4931423e+01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 3.9088322e+01,  3.4981895e+01,  0.0000000e+00, -2.7643835e+01, -6.9000001e+00, -3.9526379e+01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  3.9526379e+01, -6.5999994e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00, -8.1183472e+00, -7.4602180e+00, -1.7347816e+01,  0.0000000e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  2.6171078e+01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00, -2.5673907e-14,  -2.6171078e+01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  2.6367797e-14,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.4931423e+01, 0.0000000e+00,  0.0000000e+00,  2.6171078e+01,  0.0000000e+00, -2.8500000e+01, -4.3849845e+00,  0.0000000e+00,  0.0000000e+00, 0.0000000e+00, -8.2175179e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00, 4.3787756e+00, -8.8000002e+00,  4.4212246e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  7.9793596e+00,  0.0000000e+00,  0.0000000e+00, 0.0000000e+00, -4.4793596e+00, -3.5000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  7.3712506e+00,  0.0000000e+00,  0.0000000e+00, 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, -5.4000001e+00, -1.9712504e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00, 0.0000000e+00,  1.7057173e+01,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.9592170e+00, -1.2600000e+01, -6.4163899e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00, 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00, 8.1257477e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00, 6.2742519e+00, -1.4400000e+01]], dtype=float32),
#
# (array([ 1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13]),
#  array([1, 2, 5, 5, 7, 0]),
#  array([], dtype=int64),
#  array([ 0,  0,  1,  1,  1,  2,  3,  5,  5,  5,  8,  8,  9, 11, 12,  3,  3, 4,  6,  8]),
#  array([ 1,  4,  2,  3,  4,  3,  4, 10, 11, 12,  9, 13, 10, 12, 13,  6,  8,  5,  7,  6])))
#
# [0.34012988 0.36042336 0.2721007  0.26722595 0.8281293  0.26920095
#  0.34334618 0.5315159  0.49516448 0.7316751  0.28853893 0.3826594
#  0.28927472 0.43187788 0.39644888 0.5446019  0.5334674  0.9244734
#  0.4533958  0.4615403 ]

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






    # def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
    #     x_b = []
    #
    #     batch_size = observations['x'].shape[0]
    #     for i in range(batch_size):
    #         # Skip invalid observations
    #         if observations['x'][i].numel() == 0 or observations['edge_idx'][i].numel() == 0 or \
    #                 observations['edge_weight'][i].numel() == 0:
    #             continue
    #
    #         edge_idx, edge_weight = self._get_edge_attr(observations['edge_idx'][i], observations['edge_weight'][i])
    #         if edge_idx is None or edge_weight is None:
    #             continue
    #
    #         data = Data(x=observations['x'][i].to(dtype=th.float32),
    #                     edge_index=edge_idx.to(device=th.device("cuda"), dtype=th.int64),
    #                     edge_attr=edge_weight.to(device=th.device("cuda"), dtype=th.float32))
    #         x_b.append(self.gcn.forward(data, self._dropout))
    #
    #     if len(x_b) == 0:
    #         raise ValueError("All observations are invalid. Cannot proceed with feature extraction.")
    #
    #     x = th.stack(x_b, dim=0) if len(x_b) > 1 else x_b[0].unsqueeze(0)
    #
    #     flatten = nn.Flatten()
    #     flattened_spaces = []
    #
    #     for key, _ in observations.items():
    #         if key != "x" and key != "edge_idx" and key != "edge_weight":
    #             flattened_spaces.append(flatten(observations[key]))
    #
    #     flattened_spaces.append(flatten(x))
    #     return th.cat(flattened_spaces, dim=1)
    #
    # def _get_edge_attr(self, edge_idx, edge_weight):
    #     # Initialize edge tensors with zeros
    #     i = edge_idx.shape[1] - 1
    #
    #     while edge_idx[0][i] == -1 and i != 0:
    #         i -= 1
    #
    #     if i != 0:
    #         edge_idx = edge_idx[:, :i + 1]
    #         edge_weight = edge_weight[:i + 1]
    #     else:
    #         edge_idx = None
    #         edge_weight = None
    #
    #     return edge_idx, edge_weight
