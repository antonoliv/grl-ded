import matplotlib
import matplotlib.pyplot as plt
import grid2op
from grid2op.Agent import RandomAgent


# Create the environment
env_name = ("./data_grid2op/l2rpn_case14_sandbox/")
env = grid2op.make(env_name)

# Initialize variables
myagent = RandomAgent(env.action_space)
obs = env.reset()
reward = env.reward_range[0] 
done = False
nb_step = 0

update_interval = 1                             # Update interval
max_iter = 40                                   # Maximum number of iterations


while not done:
    fig = env.render()                          # render the environment
    plt.show()  # Show the plot after each iteration
    act = myagent.act(obs, reward, done)        # call the act method of the agent
    obs, reward, done, info = env.step(act)     # take the action and update the environment
    nb_step += 1                                # update the number of steps

    if nb_step >= max_iter:
        break
