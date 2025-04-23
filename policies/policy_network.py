from typing import List

import gymnasium as gym
import torch
import torch.nn as nn

from policies.policy import Policy


class MLP_Xavier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim=256, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(MLP_Xavier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights uniformly between -0.001 and 0.001
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain = 1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class PolicyNetwork(Policy):
    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 256,
        seed=None
    ):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.policy_net = MLP_Xavier(state_dim, action_dim, hidden_dim, seed)
        super().__init__(self.policy_net)

    def select_action(self, state: int):
        logits = self.policy_net(torch.tensor(state, dtype=torch.float32))
        prob = torch.softmax(logits, dim=-1)
        prob = torch.clamp(prob, min = 1e-8, max = 1.0)
        prob = prob / prob.sum()
        action = torch.multinomial(prob, num_samples=1).item()
        return action, prob[action]

    def get_probabilites(self, state: int):
        logits = self.policy_net(torch.tensor(state, dtype=torch.float32))
        prob = torch.softmax(logits, dim=-1)
        return prob

    def get_probabilites_states(self, states: List[int]):
        return torch.softmax(self.policy_net(states), dim=-1)
