import random

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from policies.boltzman import BoltzmannPolicy
from policies.policy import Policy

gym.register_envs(ale_py)


class MLP_Xavier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP_Xavier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights uniformly between -0.001 and 0.001
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    def __init__(
        self,
        env: gym.Env,
        policy: Policy,
        seed: int, 
        temperature_decay: bool,
        alpha_theta: float = 0.001,
        alpha_w: float = 0.001,
        gamma: float = 0.99,
    ):
        self.gamma = gamma
        self.state_dim = env.observation_space.shape[0]
        self.temperature_decay = temperature_decay
        self.env = env
        self.actor = policy
        self.seed = seed
        self.critic = MLP_Xavier(self.state_dim, 1)
        self.actor_optimizer = optim.Adam(
            self.actor.policy_net.parameters(), lr=alpha_theta
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha_w)
        self.I = 1.0

    def seed_model(self, seed):
        # Set random seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Initialize environment with seed
        if seed is not None:
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        # Compute the value estimates from the value-estimate critic network
        value = self.critic(state_tensor)
        next_value = (
            self.critic(next_state_tensor) if not done else 0.0
        )  # For the case of terminal states

        # TD error
        delta = reward_tensor + self.gamma * next_value - value

        # Update the critic (State-Value function)
        critic_loss = delta.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Policy Gradient Update
        probs = self.actor.get_policy(state)
        prob = probs[action]
        policy_loss = -torch.log(prob + 1e-8) * self.I * delta.detach()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Decay i
        self.I *= self.gamma

    def train(self, number_of_episodes=1000, max_iterations=1000):
        self.seed_model(self.seed)
        episode_rewards = []
        for _ in tqdm(range(number_of_episodes), desc="Actor-Critic running..."):
            self.I = 1.0
            reset_result = self.env.reset()  # s_0
            if isinstance(reset_result, tuple):
                state = reset_result[0]  # Extract state from (state, info) tuple
            else:
                state = reset_result

            total_reward = 0
            done = False
            truncated = False

            itr = 0
            while not (done or truncated) and itr < max_iterations:
                # Choose action based on current state
                action_pair = self.actor.select_action(state)

                # Take action in environment - newer versions return (next_state, reward, done, truncated, info)
                step_result = self.env.step(action_pair[0])
                if len(step_result) == 5:  # New gym API
                    next_state, reward, done, truncated, _ = step_result
                else:  # Old gym API
                    next_state, reward, done, _ = step_result
                    truncated = False

                self.update(
                    state, action_pair[0], reward, next_state, done or truncated
                )

                # Update state and accumulate reward
                state = next_state
                total_reward += reward
                itr += 1

                if self.temperature_decay:
                    self.actor.decay_temperature()

            episode_rewards.append(total_reward)
        return episode_rewards


# For testing purpose
def main() -> None:
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    initial_temperature = 1.0
    policy = BoltzmannPolicy(env =env,  initial_temperature=initial_temperature)
    actor_critic = ActorCritic(
        env=env,
        policy=policy,
        seed = 1,
        temperature_decay=False,
    )
    rewards = actor_critic.train()
    episodes = np.arange(len(rewards))
    mean_reward = np.mean(rewards, axis=0)
    std_reward = np.std(rewards, axis=0)

    plt.plot(
        episodes,
        rewards,
        label=f"Actor-Critic algorithm w/ No-Decay.",
        color="green",
        linestyle="--",
    )
    plt.fill_between(
        episodes,
        mean_reward - std_reward,
        mean_reward + std_reward,
        color="green",
        alpha=0.2,
    )
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
