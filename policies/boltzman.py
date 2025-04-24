import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

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


class BoltzmannPolicy(Policy):
    def __init__(
        self,
        env: gym.Env,
        initial_temperature: float,
        min_temperature: float = 0.1,
        decay_steps: int = 1000,
        seed: int = None,
    ):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.seed = seed
        if seed is not None:
            self.seed_model(seed)
        self.policy_net = MLP(state_dim, action_dim)
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optim.SGD([torch.tensor(self.temperature)], lr=initial_temperature),
            start_factor=1.0,
            end_factor=min_temperature / initial_temperature,
            total_iters=decay_steps,
        )

    def select_action(self, state: int):
        logits = self.policy_net(torch.tensor(state, dtype=torch.float32))
        prob = torch.softmax(logits / self.temperature, dim=-1)
        action = torch.multinomial(prob, num_samples=1).item()
        return action, prob[action]
    
    def seed_model(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.seed = seed

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
