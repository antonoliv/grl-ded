import os
import time
from abc import ABC

import gymnasium as gym
import numpy as np
import pandas as pd
import stable_baselines3
from stable_baselines3.common.utils import get_device
from torch_geometric.seed import seed_everything
import settings
from grl.environment import make_env
from .train_callback import TrainCallback


def set_seed(seed, path):
    """
    Get the seed for reproducibility.

    :param path: path to save the seed
    :return: seed
    """
    # SEED = 123456789

    if seed is None:
        seed = int(time.time())

    # Save the seed
    with open(path + "seed.txt", "w+", encoding="utf-8") as file:
        file.write(f"Seed: {seed}\n")

    seed_everything(seed)
    return seed


class BaseModel(ABC):
    """
    Base class for all models.
    """

    SB3Model = stable_baselines3.common.base_class.BaseAlgorithm
    GRID2OP_DATA = settings.GRID2OP_DATA
    EXPERIMENTS = settings.EXPERIMENTS
    MAX_ITER = settings.MAX_ITER

    # make_env = make_env

    def __init__(
        self,
        name: str,
        verbose: int,
        model_params: dict,
        env_params: dict,
        seed: int = int(time.time()),
    ):
        """ "
        Initialize the BaseModel object.

        :param name:                name of the model
        :param train_episodes:      number of training episodes
        :param eval_episodes:       number of evaluation episodes
        :param verbose:             verbosity level
        :param model_params:        model parameters
        :param env_params:          environment parameters
        :param seed:                seed for reproducibility
        """

        # Check attributes attributes are not none
        if verbose is None:
            raise ValueError("Verbose is empty")
        if name is None or str.strip(name) == "":
            raise ValueError("Name is empty")

        # Check environment parameters
        if "env_path" not in env_params:
            raise ValueError("Environment Name is empty")
        if "reward" not in env_params:
            raise ValueError("Reward is empty")
        if "obs_scaled" not in env_params:
            raise ValueError("Observation Scaled flag is empty")
        if "obs_step" not in env_params:
            raise ValueError("Observation Step flag is empty")
        if "act_no_curtail" not in env_params:
            raise ValueError("Action No Curtail flag is empty")

        # Set attributes
        self.device = get_device("auto")
        self.path = self.EXPERIMENTS + name + "/"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.seed = set_seed(seed, self.path)
        self.verbose = verbose
        self.name = name
        self.trained = os.path.exists(self.path + "model.zip")

        # Set model and environment parameters
        self._model_params = model_params.copy()
        self._env_kwargs = env_params.copy()

        # Set environment name
        env_name = env_params["env_path"]
        self._train_path = f"{self.GRID2OP_DATA}{env_name}_train/"
        self._val_path = f"{self.GRID2OP_DATA}{env_name}_val/"
        del self._env_kwargs["env_path"]

        # Set model class
        self.model = model_params["class"]
        del self._model_params["class"]

        # Set the curtailment limit
        match self._env_kwargs["climit_type"]:
            case None:
                self._env_kwargs["act_curtail_limit"] = 0.0
            case "fixed":
                self._env_kwargs["act_curtail_limit"] = env_params["climit_low"]

    def _setup_train(self, train_env) -> SB3Model:
        """
        Setup the model for training.

        :param train_env:       training environment
        :return:                stable-baselines3 model
        """

        # Set the environment
        model_params = self._model_params.copy()
        model_params["env"] = train_env

        return self.model(**model_params)

    def _setup_validate(self, val_env) -> SB3Model:
        """
        Setup the model for validation.

        :param val_env:         validation environment
        :return:                stable-baselines3 model
        """
        # If model is not trained raise an error
        if not self.trained:
            raise ValueError("Model not trained")

        # Load the model
        model = self.model.load(
            self.path + "model",
            env=val_env,
            seed=self.seed,
            device=self.device,
            custom_objects={
                "observation_space": val_env.observation_space,
                "action_space": val_env.action_space,
            },
        )

        return model

    def _init_env(self, env_path, env_kwargs) -> gym.Env:
        """
        Initialize the environment.
        :param env_path:    path to the environment
        :return:            environment
        """

        # Remove the curtailment limit parameters (used in the training callback only)
        del env_kwargs["climit_type"]
        del env_kwargs["climit_low"]
        del env_kwargs["climit_end"]
        del env_kwargs["climit_factor"]

        # Set the environment name and the seed
        env_kwargs["env_path"] = env_path
        env_kwargs["seed"] = self.seed

        return make_env(**env_kwargs)

    def _train_env(self) -> gym.Env:
        """
        Initialize the training environment.

        :return:   training environment
        """
        env_kwargs = self._env_kwargs.copy()
        if (
            env_kwargs["climit_type"] is not None
            and env_kwargs["climit_type"] != "fixed"
        ):
            env_kwargs["act_curtail_limit"] = 1.0
        print(self._train_path)
        return self._init_env(self._train_path, env_kwargs)

    def _val_env(self) -> gym.Env:
        """
        Initialize the validation environment.

        :return:  validation environment
        """
        env_kwargs = self._env_kwargs.copy()
        if (
            env_kwargs["climit_type"] is not None
            and env_kwargs["climit_type"] != "fixed"
        ):
            env_kwargs["act_curtail_limit"] = env_kwargs["climit_low"]
        return self._init_env(self._val_path, env_kwargs)

    def train(self, train_episodes: int):
        """
        Train the model.
        """

        if train_episodes is None or train_episodes < 0:
            raise ValueError("Train Episodes must be greater than 0")

        train_env = self._train_env()

        model = self._setup_train(train_env)  # Setup training callback
        total_timesteps = train_episodes * self.MAX_ITER  # Total timesteps

        climit_type = self._env_kwargs["climit_type"]
        climit_low = self._env_kwargs["climit_low"]
        climit_end = self._env_kwargs["climit_end"]
        climit_factor = self._env_kwargs["climit_factor"]

        # Create training callback, for metrics, curtailment limit update, and training stop
        train_callback = TrainCallback(
            path=self.path + "data/train/",
            model=model,
            verbose=self.verbose,
            max_episodes=train_episodes,
            climit_type=climit_type,
            climit_end=climit_end,
            climit_factor=climit_factor,
            climit_low=climit_low,
        )

        print("Started Training")

        # Train the agent
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=False,
            callback=train_callback,
        )

        # Save the model
        model.save(self.path + "model")
        print("Training Done")
        print()
        self.trained = True
        train_env.close()

    def validate(
        self,
        eval_episodes: int,
    ):
        """
        Validate the model.

        :return:   dictionary of metrics
        """

        if eval_episodes is None or eval_episodes < 0:
            raise ValueError("Eval Episodes must be greater than 0")

        if not self.trained:
            raise ValueError("Model not trained")

        val_env = self._val_env()

        # Load the model
        model = self._setup_validate(val_env)

        # Initialize path to save validation data
        path = self.path + "data/val/"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Get the grid2op environment
        grid_env = val_env.get_wrapper_attr("init_env")

        env = model.get_env()

        # Episode Metrics

        start_time = time.time()  # Start time
        cost = 0  # Episode cost
        res_waste = 0  # Episode renewable energy wasted
        acc_reward = 0  # Accumulated reward of episode
        length = 0  # Length of episode
        avg_cost = []  # Average cost of each episode
        avg_res_waste = []  # Average renewable energy wasted of each episode
        acc_rewards = []  # Accumulated reward of each episode
        lengths = []  # Length of each episode

        # Number of Renewable Generators
        n_res = 0
        for i in range(0, grid_env.n_gen):
            if grid_env.gen_renewable[i] == 1:
                n_res += 1

        print("Started Validation")

        for i in range(eval_episodes):

            # Reset the environment
            obs = env.reset()

            while True:
                # fig = env.render()  # render the environment
                # plt.pause(update_interval)  # Show the plot after each iteration

                # Predict and execute the action
                action, _states = model.predict(obs, deterministic=1)
                obs, reward, terminated, info = env.step(action)

                acc_reward += reward[0]
                length += 1

                cost += (
                    (obs["gen_p"] * grid_env.gen_cost_per_MW).sum()
                    * grid_env.delta_time_seconds
                    / 3600.0
                )

                for j in range(0, grid_env.n_gen):
                    if obs["gen_p_before_curtail"][0][j] != 0:
                        res_waste += (
                            obs["gen_p_before_curtail"][0][j] - obs["gen_p"][0][j]
                        ).sum()

                if terminated:
                    # when episode is over

                    acc_rewards.append(acc_reward)
                    avg_cost.append(cost * 288 / length)
                    avg_res_waste.append(res_waste * 288 / length)
                    lengths.append(length)

                    if self.verbose:
                        print(
                            f"Episode: {i + 1} \nLength: {length} \nAccumulated Reward: {acc_reward}\n"
                        )

                    acc_reward = 0
                    length = 0
                    cost = 0
                    res_waste = 0
                    break

        env.close()

        time_elapsed = time.time() - start_time
        avg_reward = np.mean(acc_rewards)
        daily_cost = np.mean(avg_cost)
        avg_res_wasted = np.mean(avg_res_waste)
        avg_len = np.mean(lengths)

        data = {
            "time": [time_elapsed],
            "episodes": [eval_episodes],
            "avg_length": [avg_len],
            "avg_reward": [avg_reward],
            "avg_cost": [daily_cost],
            "avg_res_wasted": avg_res_wasted,
        }

        df_info = pd.DataFrame(data)

        episode = {
            "episode": range(1, eval_episodes + 1),
            "acc_reward": acc_rewards,
            "length": lengths,
            "cost": avg_cost,
            "res_wasted": avg_res_waste,
        }

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
        df_info.to_csv(path + "info.csv", index=False)
        df_episode.to_csv(path + "episode.csv", index=False)

        print("Validation Done")
        print()

        val_env.close()
        return (
            avg_reward,
            daily_cost,
            avg_res_wasted,
            avg_len,
            time_elapsed,
        )

    def train_and_validate(self, train_episodes: int, eval_episodes: int):
        """
        Train and validate the model.

        :return:   dictionary of metrics
        """

        print()
        print(self.name + " - " + self.path)
        print()

        self.train(train_episodes)

        avg_reward, avg_cost, avg_res_wasted, avg_len, time_ellapsed = self.validate(
            eval_episodes
        )
        return {
            "mean_reward": avg_reward,
            "mean_cost": avg_cost,
            "mean_res_wasted": avg_res_wasted,
            "mean_length": avg_len,
            "time": time_ellapsed,
        }
