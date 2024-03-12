from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
import time
import os

class CollectCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, save_path, verbose: int = 0):
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

        self.episode_step = 0
        self.acc_reward_episode = 0
        self.acc_rewards_episodes = []
        self.start_time = time.time()

        self.steps = 0              # Number of Iterations
        self.episodes = 0           # Number of Episodes
        self.reward = []            # Reward of each step
        self.acc_reward = []        # Accumulated reward of each step
        self.episode_lengths = []   # Length of each episode

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
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # Record reward for every step
        self.reward.append(self.locals['rewards'][0])
        self.acc_reward.append(self.acc_reward[-1] + self.reward[-1] if len(self.acc_reward) > 0 else self.reward[-1])
        self.steps += 1
        self.episode_step += 1
        if self.locals['dones'][0]:
            # Episode ended, increment counter and save episode info
            self.episodes += 1
            self.episode_lengths.append(self.episode_step)
            self.episode_step = 0
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
            'Num Steps': [self.steps],
            'Num Episodes': [self.episodes],
            'Avg Steps per Episode': [np.mean(self.episode_lengths) if self.episodes > 0 else 0],
        }


        step = {
            'Iteration': range(len(self.reward)),
            'Reward': self.reward,
            'Accumulative Reward': self.acc_reward
        }


        df_info = pd.DataFrame(data)
        step_zip = list(zip(step["Iteration"], step["Reward"], step["Accumulative Reward"]))
        df_step = pd.DataFrame(step_zip, columns = list(step.keys()))
        # Save to CSV
        df_info.to_csv(self.save_path + "info.csv", index=False)
        print(f'Saved training info to {self.save_path + "info.csv"}')
        df_step.to_csv(self.save_path + "step.csv", index=False)
        print(f'Saved training info to {self.save_path + "step.csv"}')
        pass
