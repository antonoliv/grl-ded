from core import train, validate
import stable_baselines3
from stable_baselines3.common.noise import NormalActionNoise
from torch import nn
import numpy as np
import torch as th

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


def sac(path, train_env, val_env, train_ep, eval_ep, seed):
    # Instantiate the agent

    net_arch = [num_units_layer] * num_hidden_layers
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation_fn
    )

    sac_model = stable_baselines3.SAC("MultiInputPolicy",
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
                    learning_starts=100,
                    train_freq= 1,
                    action_noise=None,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    ent_coef="auto",
                    target_entropy= "auto",
                    use_sde= False,
                    sde_sample_freq= -1,
                    use_sde_at_warmup= False,
                    device=th.device("cuda"),
                    seed=seed)


    train(path, sac_model, train_ep)

    del sac_model

    sac_model = stable_baselines3.SAC.load(path + "model", env=val_env, seed=seed)

    validate(path, sac_model, val_env.get_wrapper_attr('init_env'), eval_ep)

def gcn_sac(path, train_env, val_env, train_ep, eval_ep, seed):

    from gnn.graph_extractor import GCNExtractor

    extractor_kwargs = dict(
        gnn_arch=(100, 20),
        dropout=0.5
    )

    net_arch = [num_units_layer] * num_hidden_layers

    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation_fn,
        features_extractor_class=GCNExtractor,
        features_extractor_kwargs=extractor_kwargs
    )

    # Instantiate the agent
    sac_model = stable_baselines3.SAC("MultiInputPolicy",
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
                                      device=th.device("cuda"),
                                      seed=seed)

    train(path, sac_model, train_ep)

    del sac_model

    sac_model = stable_baselines3.SAC.load(path + "model", env=val_env, seed=seed)

    validate(path, sac_model, val_env.get_wrapper_attr('init_env'), eval_ep)

def a2c():
    pass

def ppo(path, train_env, val_env, train_ep, eval_ep, seed):

    net_arch = [num_units_layer] * num_hidden_layers
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation_fn
    )



    ppo_model = stable_baselines3.PPO("MultiInputPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    clip_range_vf=None,
                    normalize_advantage=True,
                    ent_coef=0.0,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    use_sde=False,
                    sde_sample_freq=-1,
                    rollout_buffer_class= None,
                    rollout_buffer_kwargs= None,
                    target_kl= None,
                    policy_kwargs=policy_kwargs,
                    device=th.device("cuda"),
                    verbose=0, 
                    seed=seed)
    
    train(path, ppo_model, train_ep)

    del ppo_model

    ppo_model = stable_baselines3.PPO.load(path + "model", env=val_env, seed=seed)

    validate(path, ppo_model, val_env.get_wrapper_attr('init_env')('init_env'), eval_ep)

def ddpg(path, train_env, val_env, train_ep, eval_ep, seed):
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    net_arch = [num_units_layer] * num_hidden_layers
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation_fn
    )
    ddpg_model = stable_baselines3.DDPG("MultiInputPolicy",
                    train_env,
                    action_noise=action_noise,
                    learning_rate=learning_rate,
                    gradient_steps=gradient_steps,
                    tau=tau,
                    gamma=gamma,
                    batch_size=batch_size,
                    buffer_size=buffer_size,  # 1e6
                    learning_starts= 100,
                    train_freq= 1,
                    replay_buffer_class= None,
                    replay_buffer_kwargs= None,
                    optimize_memory_usage= False,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    device=th.device("cuda"),
                    seed=seed)

    train(path, ddpg_model, train_ep)

    del ddpg_model

    ddpg_model = stable_baselines3.DDPG.load(path + "model", env=val_env, seed=seed)

    validate(path, ddpg_model, val_env.get_wrapper_attr('init_env'), eval_ep)