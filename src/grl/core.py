
from callbacks.data_collect import CollectCallback
from callbacks.train_episode import EpisodeCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy



def train(path: str, model, episodes: int, seed: int):
    # Path to save the model
    

    SEED = seed

    max_episodes = episodes
    avg_episode_length = 300
    total_timesteps = max_episodes * avg_episode_length

    # Create a callbacks
    eval_callback = CollectCallback(save_path=path + "data/train/")  # Saves training data
    episode_callback = EpisodeCallback(max_episodes=max_episodes, verbose=0)  # Stops training after x episodes
    callback_lst = CallbackList([eval_callback, episode_callback])

    # Train the agent and display a progress bar
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback_lst)

    model.save(path + "model")



def validate(model):

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    print("Mean Reward: " + str(mean_reward))
    print("Std Reward: " + str(std_reward))