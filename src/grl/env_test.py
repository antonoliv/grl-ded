import matplotlib
import matplotlib.pyplot as plt
import grid2op
from grid2op.Agent import RandomAgent
from lightsim2grid import LightSimBackend


# Create the environment
env_name = "./data_grid2op/l2rpn_idf_2023/"
backend_class = LightSimBackend
env = grid2op.make(env_name, backend=backend_class())

env.seed(0)  # for reproducible experiments
episode_count = 5  # i want to make lots of episode


# total number of episode
total_episode = len(env.chronics_handler.subpaths)

# Initialize variables
agent = RandomAgent(env.action_space)
obs = env.reset()
reward = 0
done = False
total_reward = 0

update_interval = 1                             # Update interval

# and now the loop starts
for i in range(episode_count):

    ###################################
    if i % total_episode == 0:
        # I shuffle each time i need to
        env.chronics_handler.shuffle()
    ###################################

    obs = env.reset()
    # now play the episode as usual
    while True:
        fig = env.render()  # render the environment
        plt.show()  # Show the plot after each iteration
        action = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(action)
        print("Cost per MW:" + str(env.gen_cost_per_MW))
        print("Storage chargeW:" + str(env.storage_charge))
        total_reward += reward
        if done:
            # in this case the episode is over
            break
