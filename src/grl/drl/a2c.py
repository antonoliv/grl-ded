

class A2C(object):
    def __init__(self, case_path):
        # define the local file path
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
        self.GAMMA = 0.99      # discount factor
        self.EXPLORE = 1e6  # 5e5\
        self.epoch = 1e4  # 1e4
        self.nstep = 40
        self.ent_coef = 0.01
        self.vf_coef = 0.25
        self.max_grad_norm = 0.5      # Maximum gradient norm

        # setup memory
        self.buffer = deque()
        # setup environment parameters
        self.map_size = 40
        # setup training
        self.step_t = 0
        self.temp_loss = 0
        self.entro = 0
        self.total_reward = np.empty([0, 0])

    def running(self, actor, critic, test=False):
        data_all = pd.read_csv(self.reward_data_path + "reward_data.csv")
        temp_i = 0
        Test = test
        env = robot.ExplorationEnv(self.map_size, 0, Test)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = actor
        value_net = critic
        params = list(policy_net.parameters()) + list(value_net.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-5)

        temp_reward_data = []
        temp_loss_data = []
        while temp_i < self.epoch:
            self.step_t += 1
            temp_i += 1
            # get the input data (X, A)
            adjacency, featrues, globals_features, fro_size = env.graph_matrix()
            node_size = adjacency.shape[0]
            key_size = node_size - fro_size
            s_t, b_t = self.data_process([adjacency, featrues], device)
            mask = np.zeros([node_size])
            mask[-fro_size:] = 1

            # get the output reward (Y)
            all_actions = env.actions_all_goals()
            rewards = env.rewards_all_goals(all_actions)

            # choose an action
            readout_t = self.test(s_t, b_t, mask, device, policy_net).view(-1).cpu().detach().numpy()
            val = self.test(s_t, b_t, mask, device, value_net).item()

            action_index = np.random.choice(fro_size, 1, p=readout_t)[0]
            action_index = key_size + action_index

            a_t = np.zeros([node_size])
            a_t[action_index] = 1

            # choose an action
            actions = all_actions[action_index]

            # get reward
            r_t = rewards[action_index]

            # move to the next view point
            for act in actions:
                _, done, _ = env.step(act)

            # terminal for RL value calculation
            current_done = done or env.loop_clo

            # get next state
            adjacency, featrues, globals_features, fro_size1 = env.graph_matrix()
            s_t1, b_t1 = self.data_process([adjacency, featrues], device)
            mask = np.zeros([adjacency.shape[0]])
            mask[-fro_size1:] = 1

            last_value = self.test(s_t1, b_t1, mask, device, value_net).item()

            # save to buffer
            self.buffer.append((s_t, a_t, r_t, s_t1, current_done, fro_size, val))

            # training step
            if len(self.buffer) == self.nstep:
                # get the batch variables
                s_j_batch = [d[0] for d in self.buffer]
                s_j1_batch = [d[3] for d in self.buffer]
                s_j_loader = DataLoader(s_j_batch, batch_size=self.nstep)
                for batch in s_j_loader:
                    s_j_batch = batch
                r_batch = [d[2] for d in self.buffer]
                value_j = [d[6] for d in self.buffer]

                discount_rewards = []
                ret = last_value
                for i in reversed(range(len(self.buffer))):
                    terminal = self.buffer[i][4]
                    ret = r_batch[i] + self.GAMMA * ret * (1.0-terminal)
                    discount_rewards.append(ret)
                discount_rewards = discount_rewards[::-1]

                a_batch = np.array([])
                y_adv_batch = np.array([])
                mask_batch = np.array([])
                for i in range(0, len(self.buffer)):
                    action_space = self.buffer[i][5]
                    act = self.buffer[i][1]
                    a_batch = np.append(a_batch, act)
                    node_space = len(act)
                    temp_mask = np.zeros(node_space)
                    temp_mask[-action_space:] = 1
                    temp_y = np.zeros(node_space)
                    index = np.argmax(act)
                    # get policy loss
                    temp_y[index] = discount_rewards[i] - value_j[i]
                    y_adv_batch = np.append(y_adv_batch, temp_y)
                    mask_batch = np.append(mask_batch, temp_mask)

                # perform gradient step
                self.train(s_j_batch, a_batch, mask_batch, discount_rewards, y_adv_batch,
                            device, policy_net, value_net, optimizer)
                temp_loss_data.append([self.step_t, self.temp_loss])
                self.buffer.clear()

            print("TIMESTEP", self.step_t,
                  "/ Loss", self.temp_loss, "/ Entropy", self.entro,
                  "/ EXPLORED", env.status(), "/ REWARD", r_t, "/ Terminal", current_done, "\n")

            if done:
                del env
                gc.collect()
                env = robot.ExplorationEnv(self.map_size, 0, Test)
                done = False

            data_all = data_all.append({"Step": self.step_t, "Reward": r_t}, ignore_index=True)
            self.total_reward = np.append(self.total_reward, r_t)

            # save progress every 50000 iterations
            if self.step_t % 5e4 == 0:
                torch.save(policy_net.state_dict(), self.weights_path + 'MyModel.pt')
            if self.step_t > 1000:
                new_average_reward = np.average(self.total_reward[len(self.total_reward) - 1000:])
                if self.step_t % 1e2 == 0:
                    temp_reward_data.append([self.step_t, new_average_reward])

        np.savetxt(self.object_path + "temp_reward.csv", temp_reward_data, delimiter=",")
        np.savetxt(self.object_path + "temp_loss.csv", temp_loss_data, delimiter=",")
        data_all.to_csv(self.reward_data_path + "reward_data.csv", index=False)
        torch.save(policy_net.state_dict(), self.object_path + 'Model_Policy.pt')
        torch.save(value_net.state_dict(), self.object_path + 'Model_Value.pt')

    def data_process(self, data, device):
        s_a, s_x = data
        edge_index = []
        edge_attr = []
        edge_set = set()
        for a_i in range(np.shape(s_a)[0]):
            for a_j in range(np.shape(s_a)[1]):
                if (a_i, a_j) in edge_set or (a_j, a_i) in edge_set \
                        or s_a[a_i][a_j] == 0:
                    continue
                edge_index.append([a_i, a_j])
                edge_attr.append(s_a[a_i][a_j])
                if a_i != a_j:
                    edge_index.append([a_j, a_i])
                    edge_attr.append(s_a[a_j][a_i])
                edge_set.add((a_i, a_j))
                edge_set.add((a_j, a_i))
        edge_index = torch.tensor(np.transpose(edge_index), dtype=torch.long)
        x = torch.tensor(s_x, dtype=torch.float)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        state = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        batch = torch.zeros(np.shape(s_a)[0], dtype=int).to(device)
        return state, batch

    def policy_cost(self, prob, advantages, action, mask):
        prob_flat = prob.view(-1)
        advantages_flat = advantages.view(-1)
        advantages_flat = torch.masked_select(advantages_flat, mask)
        action = torch.masked_select(action, mask)
        log_prob = prob_flat.log()
        policy_loss = -torch.mul(log_prob, advantages_flat)
        policy_loss = torch.mul(policy_loss, action).sum() / self.nstep
        return policy_loss

    def value_cost(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        loss = F.mse_loss(pred_flat, target_flat)
        return loss

    def entropy_loss(self, prob):
        prob_flat = prob.view(-1).detach()
        entro = -torch.mul(prob_flat.log(), prob_flat).sum() / self.nstep
        self.entro = entro.item()
        return entro

    def train(self, data, action, mask, dis_reward, y_adv,
              device, modelA, modelC, optimizer):
        modelA.train()
        modelC.train()
        data = data.to(device)
        mask = torch.tensor(mask, dtype=bool).to(device)
        optimizer.zero_grad()
        actor_out = modelA(data, mask, batch=data.batch)
        critic_out = modelC(data, mask, batch=data.batch)
        eps = 1e-35
        actor_out = actor_out + eps
        y_adv = torch.tensor(y_adv).to(device)
        dis_reward = torch.tensor(dis_reward).to(device)
        action = torch.tensor(action).to(device)
        actor_loss = self.policy_cost(actor_out, y_adv, action, mask)
        critic_loss = self.value_cost(critic_out, dis_reward)
        entropy_loss = self.entropy_loss(actor_out)
        loss = actor_loss - entropy_loss * self.ent_coef + critic_loss * self.vf_coef
        self.temp_loss = loss.item()
        loss.backward()
        params = list(modelA.parameters()) + list(modelC.parameters())
        for param in params:
            param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
        optimizer.step()

    def test(self, data, batch, mask, device, model):
        model.eval()
        data = data.to(device)
        mask = torch.tensor(mask, dtype=bool).to(device)
        pred = model(data, mask, batch)
        return pred


if __name__ == "__main__":
    case_path = "test_case"
    training = A2C(case_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modela = Networks.PolicyGCN()
    modelc = Networks.ValueGCN()
    modela.to(device)
    modelc.to(device)
    training.running(modela, modelc)