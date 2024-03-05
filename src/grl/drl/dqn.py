import numpy as np
import os
from collections import deque
import gc
import random
import pandas as pd
from scipy.special import softmax
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader
import Networks
import envs.exploration_env as robot


class DQN(object):
    def __init__(self, case_path, model_name):
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
        self.BATCH = 64  # 64
        self.REPLAY_MEMORY = 1e4  # 1e4
        self.GAMMA = 0.99
        self.OBSERVE = 5e3  # 5e3
        self.EXPLORE = 1e6  # 5e5
        self.epoch = 1e4  # 1e4
        if model_name == "GCN":
            self.TARGET_UPDATE = 15000
        else:
            self.TARGET_UPDATE = 9000

        # exploration and exploitation trad-off parameters
        # e-greedy trad-off
        self.FINAL_EPSILON = 0  # final value of epsilon
        self.INITIAL_EPSILON = 0.9
        self.max_grad_norm = 0.5

        # setup environment parameters
        self.map_size = 40
        # setup replay memory
        self.buffer = deque()
        # setup training
        self.step_t = 0
        self.epsilon = self.INITIAL_EPSILON
        self.temp_loss = 0
        self.total_reward = np.empty([0, 0])

    def running(self, model, modelTarget, test=False):
        data_all = pd.read_csv(self.reward_data_path + "reward_data.csv")
        temp_i = 0
        Test = test
        method = "bayesian"  # bayesian, e-greedy
        env = robot.ExplorationEnv(self.map_size, 0, Test)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = model
        target_net = modelTarget
        target_net.eval()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-5)

        temp_reward_data = []
        temp_loss_data = []
        while temp_i < self.epoch:
            self.step_t += 1
            temp_i += 1
            # e-greedy scale down epsilon
            if self.epsilon > self.FINAL_EPSILON and self.step_t > self.OBSERVE:
                self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE
            # get the input data (X, A)
            adjacency, featrues, globals_features, fro_size = env.graph_matrix()
            node_size = adjacency.shape[0]
            key_size = node_size - fro_size
            s_t = self.data_process([adjacency, featrues])

            # get the output reward (Y)
            all_actions = env.actions_all_goals()
            rewards = env.rewards_all_goals(all_actions)

            # choose an action
            if method == "e-greedy":
                readout_t = self.test(s_t, 0.0, device, policy_net).cpu().detach().numpy()
                a_t = np.zeros([node_size])
                if random.random() <= self.epsilon:
                    # Random Action
                    state = "explore"
                    action_index = random.randrange(fro_size)
                    a_t[key_size + action_index] = 1
                else:
                    # Policy Action
                    state = "exploit"
                    action_index = np.argmax(readout_t[-fro_size:])
                    a_t[key_size + action_index] = 1
            elif method == "bayesian":
                readout_t = self.test(
                    s_t, self.epsilon, device, policy_net).cpu().detach().numpy()
                state = "bayesian"
                a_t = np.zeros([node_size])
                action_index = np.argmax(readout_t[-fro_size:])
                a_t[key_size + action_index] = 1

            # choose an action
            actions = all_actions[key_size + action_index]

            # get reward
            r_t = rewards[key_size + action_index]

            # move to next view point
            for act in actions:
                _, done, _ = env.step(act)

            # terminal for Q networks
            # cur_ter = env.loop_clo and r_t == 1
            current_done = done or env.loop_clo

            # get next state
            adjacency, featrues, globals_features, fro_size1 = env.graph_matrix()
            s_t1 = self.data_process([adjacency, featrues])

            # save to buffer
            self.buffer.append((s_t, a_t, r_t, s_t1, current_done, fro_size1))
            if len(self.buffer) > self.REPLAY_MEMORY:
                self.buffer.popleft()

            # training step
            if self.step_t > self.OBSERVE:
                # updata target network
                if self.step_t % self.TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # sample a minibatch to train on
                minibatch = random.sample(self.buffer, self.BATCH)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]
                s_j_loader = DataLoader(s_j_batch, batch_size=self.BATCH)
                s_j1_loader = DataLoader(s_j1_batch, batch_size=self.BATCH)
                for batch in s_j_loader:
                    s_j_batch = batch
                for batch1 in s_j1_loader:
                    s_j1_batch = batch1

                r_batch = [d[2] for d in minibatch]
                readout_j1_batch = self.test(s_j1_batch, 0.0, device, target_net)
                readout_j1_batch = readout_j1_batch.cpu().detach().numpy()
                a_batch = np.array([])
                y_batch = np.array([])
                start_p = 0
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    action_space = minibatch[i][5]
                    act = minibatch[i][1]
                    a_batch = np.append(a_batch, act)
                    node_space = len(act)

                    temp_y = np.zeros(node_space)
                    index = np.argmax(act)
                    # if terminal, only equals reward
                    if terminal:
                        temp_y[index] = r_batch[i]
                    else:
                        temp_range = readout_j1_batch[start_p:start_p + node_space]
                        temp_range = temp_range[-action_space:]
                        max_q = np.max(temp_range)
                        temp_y[index] = r_batch[i] + self.GAMMA * max_q
                    start_p += node_space
                    y_batch = np.append(y_batch, temp_y)

                # perform gradient step
                self.train(s_j_batch, a_batch, y_batch, device, policy_net, optimizer)
                temp_loss_data.append([self.step_t, self.temp_loss])

            print("TIMESTEP", self.step_t, "/ STATE", state, "/ EPSILON", self.epsilon,
                  "/ Q_MAX %e" % np.max(readout_t), "/ EXPLORED", env.status(), "/ REWARD", r_t,
                  "/ Terminal", current_done, "\n")

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
        torch.save(target_net.state_dict(), self.object_path + 'Model_Target.pt')

    def data_process(self, data):
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
        return state

    def cost(self, pred, target, action):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        readout_action = torch.mul(pred_flat, action)
        loss = torch.pow(readout_action - target_flat, 2).sum() / self.BATCH
        return loss

    def train(self, data, action, y, device, model, optimizer):
        model.train()
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data, 0.5, batch=data.batch)
        y = torch.tensor(y).to(device)
        action = torch.tensor(action).to(device)
        loss = self.cost(out, y, action)
        self.temp_loss = loss.item()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
        optimizer.step()

    def test(self, data, prob, device, model):
        model.eval()
        data = data.to(device)
        pred = model(data, prob)
        return pred


