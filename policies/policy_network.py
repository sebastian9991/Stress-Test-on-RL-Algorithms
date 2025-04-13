from typing import List

import torch
import torch.nn as nn

from policies.policy import Policy


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights uniformly between -0.001 and 0.001
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.001, 0.001)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class PolicyNetwork(Policy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        self.policy_net = MLP(state_dim, action_dim, hidden_dim)
        super().__init__(self.policy_net)

    def select_action(self, state: int):
        logits = self.policy_net(torch.tensor(state, dtype=torch.float32))
        prob = torch.softmax(logits, dim=-1)
        action = torch.multinomial(prob, num_samples=1).item()
        return action, prob[action]

    def get_probabilites(self, state: int):
        logits = self.policy_net(torch.tensor(state, dtype=torch.float32))
        prob = torch.softmax(logits, dim=-1)
        return prob

    def get_probabilites_states(self, states: List[int]):
        return torch.stack(
            [
                torch.softmax(
                    self.policy_net(torch.tensor(state, dtype=torch.float32)), dim=-1
                )
                for state in states
            ]
        )
