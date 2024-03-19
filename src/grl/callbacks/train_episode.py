from stable_baselines3.common.callbacks import BaseCallback


class EpisodeCallback(BaseCallback):
    def __init__(self, max_episodes, verbose=0):
        super(EpisodeCallback, self).__init__(verbose)
        self.max_episodes = max_episodes
        self.current_episodes = 0

    def _on_step(self) -> bool:
        if 'episode' in self.locals:
            if self.locals['done']:
                self.current_episodes += 1
                if self.verbose > 0:
                    print("Episode: ", self.current_episodes)
                if self.current_episodes >= self.max_episodes:
                    return False  # Return False to stop the training
        return True
