# Remember to adjust your student ID in meta.xml
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchPolicy(nn.Module):
    def __init__(self, state_size, action_size, lr=1e-4):
        super(PyTorchPolicy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, self.action_size)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.output_layer(x), dim=-1)
        return action_probs

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

def get_agent_state(obs):
    taxi_row, taxi_col, sta_0_x, sta_0_y, sta_1_x, sta_1_y, sta_2_x, sta_2_y, sta_3_x, sta_3_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    state = torch.tensor([taxi_row, taxi_col, sta_0_x, sta_0_y, sta_1_x, sta_1_y, sta_2_x, sta_2_y, sta_3_x, sta_3_y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look])
    state = state.float()
    return state

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    policy_model = PyTorchPolicy(16, 6, lr=0.0001)
    policy_model.load("policy_model.pth")
    state = get_agent_state(obs)
    action, _ = policy_model.get_action(state)
    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

