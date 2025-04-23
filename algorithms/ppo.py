import random
from typing import Dict, List

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from policies.policy import Policy
from policies.policy_network import PolicyNetwork
from utils.helper_gradient import (conjugate_gradient, flat_grad,
                                   kl_div_categorical)

gym.register_envs(ale_py)



class fake_action_space:
    def __init__(self, n):
        self.n = n

class fake_observation_space:
    def __init__(self, shape):
        self.shape = shape

class fake_env:
    def __init__(self):
        self.state = 0
        self.action_space = fake_action_space(2)
        self.observation_space = fake_observation_space([1])

    def reset(self, seed=0):
        self.state = 0
        return np.array([self.state]), None

    def step(self, action):
        if action == 1:
            self.state += 1
        else:
            self.state = max(0, self.state-1)

        terminated = False
        if self.state == 3:
            terminated = True

        if terminated:
            reward = 0
        else:
            reward = -1
        return np.array([self.state]), reward, terminated, False, None


class PPO:
    def __init__(
        self,
        env: gym.Env,
        policy: Policy,
        seed: int,
        do_stress_test: bool = False,
        epsilon: float = 0.1,
        n_iter_conv: int = 5,
        step_size: float = 0.001
    ):
        self.epsilon = epsilon
        self.K = n_iter_conv
        self.env = env
        self.policy = policy
        self.seed = seed
        self.do_stress_test = do_stress_test
        self.step_size = step_size

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

    def get_flat_params(self):
        """Flatten all parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in self.policy.policy_net.parameters()])

    def set_flat_params(self, flat_params) -> None:
        """Set all parameters from a flattened vector."""
        idx = 0
        for p in self.policy.policy_net.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[idx : idx + numel].view(p.shape))
            idx += numel

    def run_episode(self, max_episode_length):
        states, actions, rewards = [], [], []
        state = self.env.reset()  # We let s_0 approx p_0 be defined by the env
        if isinstance(state, tuple):
            state = state[0]

        eps = 0
        while eps < max_episode_length:
            action_pair = self.policy.select_action(state)
            action = action_pair[0]

            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, truncated, _ = step_result
            else:
                next_state, reward, done, _ = step_result
                truncated = False

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

            if done:
                break
            eps += 1

        return states, actions, rewards

    def compute_q_values_single_path(self, states, actions, rewards, gamma=0.99):
        G = 0
        q_values = []
        for t in reversed(range(len(states))):
            G = rewards[t] + gamma * G
            q_values.insert(0, G)
        return np.asarray(q_values)

    def surrogate_loss(self, states, actions, q_values, old_probabilities):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        q_values = torch.tensor(q_values, dtype=torch.float32).detach()

        losses = []
        action_probs = self.policy.get_probabilites_states(states)
        for i in range(len(actions)):
            q = q_values[i]
            r = action_probs[i][actions[i]]/old_probabilities[i][actions[i]]
            losses.append(-torch.min(r * q, torch.clip(r, 1 - self.epsilon, 1 + self.epsilon) * q))
        loss = sum(losses) / len(losses)

        return loss

    def update_policy(self, grad_flatten) -> None:
        n = 0
        for p in self.policy.policy_net.parameters():
            numel = p.numel()
            g = grad_flatten[n : n + numel].view(p.shape)
            p.data += g
            n += numel

    def update_agent(self, states, actions, rewards, q_values, delta=0.01):
        old_probabilites = self.policy.get_probabilites(states).detach()

        for i in range(self.K):
            loss = self.surrogate_loss(states, actions, q_values, old_probabilites)
            parameters = list(self.policy.policy_net.parameters())
            g = flat_grad(loss, parameters, create_graph=True)*self.step_size
            self.update_policy(g)

    def train(
        self, stress_config: Dict, number_of_episodes=1000, max_iterations=1000
    ) -> List[float]:

        self.seed_model(self.seed)
        total_rewards = []
        for eps in tqdm(range(number_of_episodes), desc="TRPO running..."):
            if self.do_stress_test and (eps == 500):
                print(f"Stress Test called at episode: {eps}")
                self.env.stress_test(**stress_config)
            states, actions, rewards = self.run_episode(
                max_episode_length=max_iterations
            )
            q_values = self.compute_q_values_single_path(states, actions, rewards)
            q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-8)
            self.update_agent(states, actions, rewards, q_values)
            sum_rewards = np.sum(rewards)
            total_rewards.append(sum_rewards)
            print(sum_rewards)

        return total_rewards


# For testing purpose
def main() -> None:
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    policy = PolicyNetwork(env=env)
    trpo = PPO(env=env, seed=1, policy=policy, epsilon=0.2, n_iter_conv=20, step_size=0.02)
    rewards = trpo.train({})
    episodes = np.arange(len(rewards))
    mean_reward = np.mean(rewards, axis=0)
    std_reward = np.std(rewards, axis=0)

    plt.plot(
        episodes,
        rewards,
        label=f"PPO algorithm",
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
