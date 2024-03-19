from grid2op.Agent import BaseAgent


class MyCustomAgent(BaseAgent):
    def __init__(self, action_space, something_else, and_another_something):
        # define here the constructor of your agent
        # here we say our agent needs "something_else" and "and_another_something"
        # to be built just to demonstrate it does not cause any problem to extend the
        # construction of the base class BaseAgent that only takes "action_space" as a constructor
        BaseAgent.__init__(self, action_space)
        self.something_else = something_else
        self.and_another_something = and_another_something

    def act(obs, reward, done=False):
        # this is the only method you need to implement
        # it takes an observation obs (and a reward and a flag)
        # and should return a valid action
        dictionary_describing_the_action = {}  # this can be anything you want that grid2op understands
        my_action = env.action_space(dictionary_describing_the_action)
        return my_action
