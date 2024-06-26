import os
import time
from typing import Optional, Dict

import numpy as np
import pandas as pd
from ray import tune

from models.sac import GCN_SAC


class Experiment(tune.Trainable):

    def setup(self, config):
        # self.name = config["name"]
        self.path = config["path"] + self.trial_name + "/"
        self.seed = config["seed"]
        # self.model = stable_baselines3.SAC
        self.config = config

        self.model = GCN_SAC(self.path, self.seed, self.trial_name)

        # gnn = GCN(
        #     in_channels=config['gnn_in_channels'],
        #     hidden_channels=config['gnn_hidden_channels'],
        #     out_channels=config['gnn_out_channels'],
        #     num_layers=config['gnn_num_layers'],
        #     dropout=config['gnn_dropout']
        # ).to(device=self.device)
        #
        # reward = RESBonusReward(res_bonus=config['res_bonus'])
        #
        # self.gnn = th.compile(gnn)
        #
        # self.train_env = env_graph(train_name, reward, self.seed, self.gnn)
        # self.val_env = env_graph(val_name, reward, self.seed, self.gnn)
        # extractor = GraphExtractor
        # extractor_kwargs = dict(
        #     ignored_keys=["step", "gen_p", "gen_p_before_curtail"]
        # )
        #
        # net_arch = [config['num_units_layer']] * config['num_hidden_layers']  # Network Architecture
        # policy_kwargs = dict(
        #     net_arch=net_arch,
        #     activation_fn=config['activation_fn'],
        #     optimizer_class=config['optimizer'],
        #     features_extractor_class=extractor,
        #     features_extractor_kwargs=extractor_kwargs
        # )

        # self.model = stable_baselines3.SAC("MultiInputPolicy",
        #                        self.train_env,
        #                        learning_rate=config["learning_rate"],
        #                        verbose=self.verbose,
        #                        gamma=config["gamma"],
        #                        use_sde=config["use_sde"],
        #                        ent_coef=config["ent_coef"],
        #                        policy_kwargs=policy_kwargs,
        #                        sde_sample_freq=config["sde_sample_freq"],
        #                        device=self.device,
        #                        seed=self.seed,
        #                        gradient_steps=config["gradient_steps"],
        #                        buffer_size=config["buffer_size"],
        #                        batch_size=config["batch_size"],
        #                        tau=config["tau"],
        #                        target_update_interval=config["target_update_interval"],
        #                        learning_starts=config["learning_starts"],
        #                        train_freq=config["train_freq"],
        #                        action_noise=None,
        #                        replay_buffer_class=None,
        #                        replay_buffer_kwargs=None,
        #                        optimize_memory_usage=False,
        #                        target_entropy="auto",
        #                        use_sde_at_warmup=False)

    def train(self):
        # with self.train_env:
        #     max_episodes = self.train_ep
        #     avg_episode_length = 1500
        #     total_timesteps = max_episodes * avg_episode_length
        #
        #     # Create a callbacks
        #     eval_callback = TrainCallback(path=self.path + "data/train/", model=self.model, verbose=1,
        #                                     max_episodes=max_episodes)  # Saves training data
        #
        #     print("Started Training")
        #     # Train the agent and display a progress bar
        #     self.model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=eval_callback)
        #     self.model.save(self.path + "model")
        #     del self.model
        #     self.model = stable_baselines3.SAC.load(self.path + "model", env=self.val_env, device=self.device)
        #     avg_reward, avg_cost, avg_res_wasted = self.validate()
        #     return {
        #         "mean_reward": avg_reward,
        #         "mean_cost": avg_cost,
        #         "mean_res_wasted": avg_res_wasted
        #     }
        return dict(self.model.train_and_validate(self.config), training_iterations=1)

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict]:
        pass

    def cleanup(self):
        pass

    def validate(self):
        with self.val_env:

            # model = self.model.load(path + "model", env=val_env, seed=self.seed)
            self.model.set_env(self.val_env)
            path = self.path + "data/val/"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            grid_env = self.val_env.get_wrapper_attr("init_env")
            # Evaluate the agent
            # NOTE: If you use wrappers with your environment that modify rewards,
            #       this will be reflected here. To evaluate with original rewards,
            #       wrap environment in a "Monitor" wrapper before other wrappers.

            env = self.model.get_env()

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
            episode_count = self.eval_ep  # i want to make lots of episode
            update_interval = 0.1  # Update interval

            print("Started Validation")

            # and now the loop starts
            for i in range(episode_count):

                obs = env.reset()

                # now play the episode as usual
                while True:
                    # fig = env.render()  # render the environment
                    # plt.pause(update_interval)  # Show the plot after each iteration
                    action, _states = self.model.predict(obs, deterministic=1)
                    obs, reward, terminated, info = env.step(action)

                    acc_reward += reward[0]
                    length += 1
                    total_steps += 1

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
                        # in this case the episode is over
                        acc_rewards.append(acc_reward)
                        avg_cost.append(cost * 288 / length)
                        avg_res_waste.append(res_waste * 288 / length)
                        lengths.append(length)
                        acc_reward = 0
                        length = 0
                        cost = 0
                        res_waste = 0
                        break

            env.close()

            data = {
                "Time Elapsed": [time.time() - start_time],
                "Num Steps": [total_steps],
                "Num Episodes": [self.eval_ep],
                "Avg Steps per Episode": [np.mean(lengths) if self.eval_ep > 0 else 0],
            }

            episode = {
                "Episode": range(1, self.eval_ep + 1),
                "Accumulative Reward": acc_rewards,
                "Length": lengths,
                "Avg Cost": avg_cost,
                "Avg Renewables Wasted": avg_res_waste,
            }

            df_info = pd.DataFrame(data)
            episode_zip = list(
                zip(
                    episode["Episode"],
                    episode["Accumulative Reward"],
                    episode["Length"],
                    episode["Avg Cost"],
                    episode["Avg Renewables Wasted"],
                )
            )
            df_episode = pd.DataFrame(episode_zip, columns=list(episode.keys()))
            # Save to CSV
            df_info.to_csv(path + "info.csv", index=False)
            df_episode.to_csv(path + "episode.csv", index=False)
            return (
                df_episode["Accumulative Reward"].mean(),
                df_episode["Avg Cost"].mean(),
                df_episode["Avg Renewables Wasted"].mean(),
            )

    def train_and_validate(self, config):

        path = self.path + "t" + str(self.exp) + "/"
        self.exp += 1
        print()
        print(self.name + " - " + path)
        print()
        env_name = config["env_path"]
        train_ep = config["train_episodes"]
        eval_ep = config["eval_episodes"]

        train_name = (
            f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_train/"
        )
        val_name = (
            f"/home/treeman/school/dissertation/src/grl/data_grid2op/{env_name}_val/"
        )

        self.train(train_name, config, train_ep, path)

        avg_reward, avg_cost, avg_res_wasted = self.validate(
            val_name, config, eval_ep, path
        )
        return {"mean_reward": avg_reward}
