import os
import time

import numpy as np
import pandas as pd

from stable_baselines3.common.callbacks import BaseCallback


class CollectCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, save_path, model, verbose: int = 0, max_episodes: int = 1000):
        super().__init__(verbose)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.save_path = save_path
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        # Global Metrics
        self.start_time = time.time()
        self.total_steps = 0  # Number of Iterations
        self.episodes = 0  # Number of Episodes
        self.model = model
        self.max_episodes = max_episodes

        self.env = self.training_env.envs[0].get_wrapper_attr('init_env')

        self.n_res = 0
        for i in range(0, self.env.n_gen):
            if self.env.gen_renewable[i] == 1:
                self.n_res += 1

        self.verbose = verbose
        # Episode Metrics
        self.length = 0
        self.acc_reward = 0
        self.cost = 0
        self.res_waste = 0
        self.avg_cost = []
        self.avg_res_waste = []
        self.acc_rewards = []  # Accumulated reward of each episode
        self.lengths = []  # Length of each episode

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        self.total_steps += 1

        if (hasattr(self.model, "learning_starts") and self.model.learning_starts >= self.total_steps):
            return True

        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        obs = self.locals['new_obs']

        # Record reward for every step
        self.acc_reward += self.locals['rewards'][0]
        self.cost += (obs['gen_p'] * self.env.gen_cost_per_MW).sum() * self.env.delta_time_seconds / 3600.0

        for i in range(0, self.env.n_gen):
            if obs['gen_p_before_curtail'][0][i] != 0:
                self.res_waste += (obs['gen_p_before_curtail'][0][i] - obs['gen_p'][0][i]).sum()

        self.length += 1

        if self.locals['dones'][0]:
            # Episode ended, increment counter and save episode info
            self.episodes += 1
            self.lengths.append(self.length)
            self.acc_rewards.append(self.acc_reward)
            self.avg_cost.append(self.cost * 288 / self.length)
            self.avg_res_waste.append(self.res_waste * 288 / self.length)

            if (self.verbose == 1):
                print("Episode: {} \nLength: {} \nAccumulated Reward: {}\n".format(
                    self.episodes, self.length, self.acc_reward))

            self.acc_reward = 0
            self.length = 0
            self.cost = 0
            self.res_waste = 0

            if self.episodes >= self.max_episodes:
                print()
                return False  # Return False to stop the training

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """

        # Prepare data for CSV
        data = {
            'Time Elapsed': [time.time() - self.start_time],
            'Num Steps': [self.total_steps],
            'Num Episodes': [self.episodes],
            'Avg Steps per Episode': [np.mean(self.lengths) if self.episodes > 0 else 0],
        }

        episode = {
            'Episode': range(1, self.episodes + 1),
            'Accumulative Reward': self.acc_rewards,
            'Length': self.lengths,
            'Avg Cost': self.avg_cost,
            'Avg Renewables Wasted': self.avg_res_waste,
        }

        df_info = pd.DataFrame(data)
        episode_zip = list(
            zip(episode["Episode"], episode["Accumulative Reward"], episode["Length"], episode["Avg Cost"],
                episode["Avg Renewables Wasted"]))
        df_episode = pd.DataFrame(episode_zip, columns=list(episode.keys()))
        # Save to CSV
        df_info.to_csv(self.save_path + "info.csv", index=False)
        df_episode.to_csv(self.save_path + "episode.csv", index=False)
        pass
