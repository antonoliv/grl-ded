import os
import time

import numpy as np
import pandas as pd
from grl.environment.action.no_curtail_action_space import NoCurtailActionSpace
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback


class TrainCallback(BaseCallback):
    """
    Callback for training the model.

    Main functionalities:
    - Record episode metrics
    - Save episode metrics to CSV
    - Update curtailment limit
    """

    def __init__(
        self,
        path: str,
        model: BaseAlgorithm,
        verbose: int,
        max_episodes: int,
        climit_type: str,
        climit_low: float,
        climit_end: int,
        climit_factor: float,
    ):
        """
        Init function for TrainCallback.

        :param path:                path to save training data
        :param model:               sb3 model
        :param verbose:             if true callback is verbose
        :param max_episodes:        max number of episodes to train
        :param climit_type:         type of curtailment limit update
        :param climit_low:          lower bound of curtailment limit
        :param climit_end:          end episode for curtailment limit update
        :param climit_factor:       factor for curtailment limit update
        """

        super().__init__(verbose)

        # Verify existence of path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if path is None or str.strip(path) == "":
            raise ValueError("Path is empty")
        if model is None:
            raise ValueError("Model is empty")
        if max_episodes is None or max_episodes <= 0:
            raise ValueError("Max Episodes must be greater than 0")
        if climit_type not in ["linear", "sqrt", "fixed", None]:
            raise ValueError("Invalid Curtailment Limit Type")
        if climit_type is not None:

            if climit_low is None or climit_low < 0.0 or climit_low > 1.0:
                raise ValueError("Curtailment Low must be between 0 and 1")

            if climit_type != "fixed":
                if climit_end is None or climit_end > max_episodes:
                    raise ValueError(
                        "Curtailment End must be greater than Max Episodes"
                    )
                if climit_factor is None or climit_factor <= 0.0:
                    raise ValueError("Curtailment Factor must be greater than 0")

        self.path = path  # Path to save training data

        # Initialize parameters
        self.model = model  # SB3 model
        self.max_episodes = max_episodes  # Max number of episodes
        self.verbose = verbose  # Verbosity
        self.env = self.training_env.envs[0].get_wrapper_attr(
            "init_env"
        )  # Grid2op Environment



        self.climit_type = climit_type  # Type of curtailment limit update
        self.climit_end = climit_end  # End episode for curtailment limit update
        self.climit_low = climit_low  # Lower bound of curtailment limit
        self.climit_factor = climit_factor  # Factor for curtailment limit update
        self.climit_done = False  # Flag to check if climit update is done

        # Curtailment Limit
        if isinstance(self.training_env.action_space, NoCurtailActionSpace):
            self.climit_type = None

        # Global Metrics
        self.start_time = time.time()  # Start Time
        self.total_steps = 0  # Total number of Iterations
        self.episodes = 0  # Total number of Episodes
        self.n_res = 0  # Number of Renewable Generators

        for i in range(0, self.env.n_gen):
            if self.env.gen_renewable[i] == 1:
                self.n_res += 1

        # Episode Metrics
        self.length = 0  # Length of Episode
        self.acc_reward = 0  # Accumulated Reward of Episode
        self.cost = 0  # Conventional Production Cost of Episode
        self.res_waste = 0  # Renewable Energy Wasted of Episode

        self.avg_cost = []  # Average Cost of each Episode
        self.avg_res_waste = []  # Average Renewable Energy Wasted of each Episode
        self.acc_rewards = []  # Accumulated reward of each episode
        self.lengths = []  # Length of each episode

        # Initial update of climit
        # self._update_climit()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: true if training should continue
        """

        # Update Steps
        self.total_steps += 1

        if (
            hasattr(self.model, "learning_starts")
            and self.model.learning_starts >= self.total_steps
        ):
            # If model has not learned enough, return True and do nothing
            return True

        # Get observation
        obs = self.locals["new_obs"]

        # Record metrics for every step
        self.acc_reward += self.locals["rewards"][0]
        self.length += 1
        self.cost += (
            (obs["gen_p"] * self.env.gen_cost_per_MW).sum()
            * self.env.delta_time_seconds
            / 3600.0
        )

        for i in range(0, self.env.n_gen):
            if obs["gen_p_before_curtail"][0][i] != 0:
                self.res_waste += (
                    obs["gen_p_before_curtail"][0][i] - obs["gen_p"][0][i]
                ).sum()

        if self.locals["dones"][0]:
            # If episode is done, record metrics

            self.episodes += 1
            self._update_climit()
            # tune.report(training_iterations=self.episodes)
            self.lengths.append(self.length)
            self.acc_rewards.append(self.acc_reward)

            # TODO put only cost and res_waste in the csv
            self.avg_cost.append(self.cost * 288 / self.length)
            self.avg_res_waste.append(self.res_waste * 288 / self.length)

            if self.verbose == 1:
                print(
                    f"Episode: {self.episodes} \nLength: {self.length} \nAccumulated Reward: {self.acc_reward}\n"
                )

            self.acc_reward = 0
            self.length = 0
            self.cost = 0
            self.res_waste = 0

            if self.episodes >= self.max_episodes:
                # if max episodes reached, return False to stop the training
                print()
                return False

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """

        # Prepare data for CSV
        data = {
            "avg_reward": [np.mean(self.acc_rewards) if self.episodes > 0 else 0],
            "avg_length": [np.mean(self.lengths) if self.episodes > 0 else 0],
            "avg_cost": [np.mean(self.avg_cost) if self.episodes > 0 else 0],
            "avg_res_wasted": [np.mean(self.avg_res_waste) if self.episodes > 0 else 0],
            "time": [time.time() - self.start_time],
            "episodes": [self.episodes],
            "total_steps": [self.total_steps],
        }

        episode = {
            "episode": range(1, self.episodes + 1),
            "acc_reward": self.acc_rewards,
            "length": self.lengths,
            "cost": self.avg_cost,
            "res_wasted": self.avg_res_waste,
        }

        df_info = pd.DataFrame(data)

        episode_zip = list(
            zip(
                episode["episode"],
                episode["acc_reward"],
                episode["length"],
                episode["cost"],
                episode["res_wasted"],
            )
        )

        df_episode = pd.DataFrame(episode_zip, columns=list(episode.keys()))

        # Save to CSV
        df_info.to_csv(self.path + "info.csv", index=False)
        df_episode.to_csv(self.path + "episode.csv", index=False)

    def _update_climit(self) -> None:
        """
        Update the curtailment limit based on the episode number.
        """
        climit = 0.0
        if self.climit_type is None or self.climit_type == "fixed" or self.climit_done:
            # If climit_type is None or fixed, do nothing
            return

        if self.climit_end >= self.episodes:
            # if climit_end is not reached, update climit

            if self.climit_type == "linear":
                # Linear
                climit = 1.0 - (1.0 - self.climit_low) * self.episodes / self.climit_end
            elif self.climit_type == "sqrt":
                # Nth Root
                climit = (1.0 - self.climit_low) * np.power(
                    ((self.climit_end - self.episodes) / self.climit_end),
                    (1 / self.climit_factor),
                ) + self.climit_low

        else:
            # If climit_end is reached, set climit to climit_low and mark the update as done
            climit = self.climit_low
            self.climit_done = True

        # Update the curtailment limit
        self.training_env.action_space.update_curtail_limit(climit)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
