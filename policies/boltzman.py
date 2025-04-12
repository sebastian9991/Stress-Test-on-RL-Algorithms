import torch
import torch.nn as nn
import torch.optim as optim
from policy import Policy


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
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


class BoltzmannPolicy(Policy):
    def __init__(
        self,
        state_dim,
        action_dim,
        initial_temperature,
        min_temperature=0.1,
        decay_steps=1000,
    ):
        self.policy_net = MLP(state_dim, action_dim)
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optim.SGD([torch.tensor(self.temperature)], lr=initial_temperature),
            start_factor=1.0,
            end_factor=min_temperature / initial_temperature,
            total_iters=decay_steps,
        )

    def select_action(self, state):
        logits = self.policy_net(torch.tensor(state, dtype=torch.float32))
        prob = torch.softmax(logits / self.temperature, dim=-1)
        action = torch.multinomial(prob, num_samples=1).item()
        return action, prob[action]

    def decay_temperature(self):
        self.scheduler.step()
        self.temperature = max(
            self.min_temperature, self.scheduler.optimizer.param_groups[0]["lr"]
        )

    # TODO: Check this
    # Most likely don't need these helper functions
    def get_policy(self, state):
        logits = self.policy_net(torch.tensor(state, dtype=torch.float32))
        return torch.softmax(
            logits - logits.max() / self.temperature, dim=-1
        )  # Numerically stable softmax

    def get_policies(self, states):
        return torch.stack(
            [
                torch.softmax(
                    self.policy_net(torch.tensor(state, dtype=torch.float32))
                    / self.temperature,
                    dim=-1,
                )
                for state in states
            ]
        )
