
class DDPG(object):
    def __init__(self, case_path):
        self.case_path = case_path
        self.weights_path = "../data/torch_weights/" + self.case_path
        self.reward_data_path = "../data/reward_data/" + self.case_path
        self.object_path = '../data/training_object_data/' + self.case_path
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        if not os.path.exists(self.reward_data_path):
            os.makedirs(self.reward_data_path)
        if not os.path.exists(self.object_path):
            os.makedirs(self.object_path)
        data_all = pd.DataFrame({"Step": [], "Reward": []})
        data_all.to_csv(self.reward_data_path + "reward_data.csv", index=False)

        # setup parameters for RL