from typing import Optional, Dict

from ray import tune


class Experiment(tune.Trainable):

    def setup(self, config):
        # self.name = params["name"]
        name = config["name"] + "/" + self.trial_name
        seed = config["seed"]
        verbose = config["verbose"]
        self.train_ep = config["train_episodes"]
        self.eval_ep = config["eval_episodes"]

        self.config = config
        sac_params = config['sac_params'].copy()
        env_params = config['env_params'].copy()
        gnn_params = config['gnn_params'].copy()

        model_class = config['class']
        self.model = model_class(
            seed,
            name,
            verbose,
            sac_params,
            env_params,
            gnn_params,
        )

    def train(self):

        return dict(self.model.train_and_validate(self.train_ep, self.eval_ep), training_iteration=1)

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Dict]:
        pass

    def cleanup(self):
        pass

