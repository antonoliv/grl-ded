import torch as th

from environment.create_env import env_test_val
from environment.observation_space import get_obs_keys
from environment.reward.res_reward import DynamicEconomicReward

path = "/experiments/"

# Environment Name
env_name = "l2rpn_case14_sandbox"
# env_name = "l2rpn_idf_2023"

# Environment paths
train_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
val_name = f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"

train_ep = 1000  # Number of Episodes
eval_ep = 100  # Number of Evaluations

test_path = path + "grl1/"

# Paths for the experiments
sac_path = "sac/"
gcn_sac_path = "gcn_sac/"

# Split the dataset
# split_dataset(env_name, SEED)

# Get the observation attributes
default_attr = get_obs_keys(False, False, False, False, False)

reward = DynamicEconomicReward(res_penalty=0.4)

SEED = 234523455
# SEED = seed(test_path)


train_env, val_env = env_test_val(train_name, val_name, default_attr, reward, SEED, False)

grid_env = train_env.init_env
observation = grid_env.reset()

weight_matrix = th.full(size=(1, 1), fill_value=-1, dtype=th.float32)

for line in range(20):
    node1 = observation.line_or_to_subid[line]
    node2 = observation.line_ex_to_subid[line]

edge_idx = (weight_matrix > -1).nonzero(as_tuple=False).t().contiguous()
edge_weight = weight_matrix[weight_matrix > -1]

print(edge_idx)

print(edge_weight)
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
