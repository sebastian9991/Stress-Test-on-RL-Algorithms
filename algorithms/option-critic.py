
from hmac import new
from tokenize import Double
from collections import deque
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.optim.rmsprop
from tqdm import tqdm
import math
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

class ActorCritic:
    def __init__(self, env: gym.Env, lr: float, lr_v: float, gamma: float = 1, T=0.1, T_decay=None, device='cpu', seed: int = 23, input_scale=1.0):
        self.env = env
        self.lr = lr
        self.lr_v = lr_v
        self.seed = seed
        self.device = device
        self.seed_model(seed)
        self.gamma = gamma  # TODO
        self.T = T
        self.T_decay = T_decay
        self.input_scale = input_scale

        # We have 1d observations for this assignment, but adding more for more general case
        self.observation_size = 1
        for x in env.observation_space.shape:
            self.observation_size = self.observation_size * x

        self.actions = list(range(env.action_space.n))
        self.z = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, env.action_space.n)
        )
        self.v = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

        self.reinit_weights()

        self.z.to(device)
        self.optimizer = torch.optim.RMSprop(self.z.parameters(), lr=self.lr)
        self.optimizer_v = torch.optim.RMSprop(self.v.parameters(), lr=self.lr_v)

    def seed_model(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def plot_reward(self, rewards: list) -> None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        # Plot the rewards

        axs[0, 0].plot(rewards)
        axs[0, 0].set_title("Reward")
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].set_ylabel("Reward")

        # Plot the cumulative reward
        cumulative = np.cumsum(rewards)
        axs[1, 0].plot(cumulative)
        axs[1, 0].set_title("Cumulative Reward")
        axs[1, 0].set_xlabel("Episode")
        axs[1, 0].set_ylabel("Reward")

        # Plot the moving average
        moving_average = []
        for i in range(len(rewards)):
            moving_average.append(np.mean(rewards[max(0, i - 5):min(len(rewards), i + 5)]))
        axs[0, 1].plot(moving_average)
        axs[0, 1].set_title("Moving Average")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Reward")

        # Space out the subplots by a bit
        plt.tight_layout()
        plt.show()

    def reinit_weights(self) -> None:
        for layer in self.z:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.uniform_(layer.weight, -0.001, 0.001)
                torch.nn.init.uniform_(layer.bias, -0.001, 0.001)

    def do_episode(self, episode_len, episode):

        state, _ = self.env.reset(
            seed=self.seed + episode)  # ADD epsiode so the seed is different for each episode

        states = []
        actions = []
        log_probs = []
        rewards = []
        t = 0
        while t < episode_len:
            output = self.z.forward(torch.from_numpy(state/self.input_scale).float().to(self.device)).squeeze()

            if self.T_decay is None:
                probs = torch.nn.functional.softmax(output/(self.T))
            else:
                probs = torch.nn.functional.softmax(output/(self.T*self.T_decay**episode))

            action = np.random.choice(self.actions, p=probs.detach().cpu().numpy())

            log_prob = torch.log(probs[action])

            # print("Action off device: ", action)
            new_state, reward, terminated, truncated, info = self.env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            t += 1
            state = new_state
            if terminated or truncated:
                break

        return [states, actions, log_probs, rewards]

    def train(self, num_episodes: int, episode_len: int, plot_results=False):
        # Collect episode
        # update replay buffer if you have one
        # update the Neural network
        # Replay and batch size only used if use_buffer is True

        self.seed_model(self.seed)
        total_rewards_v = []
        for episode in tqdm(range(num_episodes), leave=False, desc="Episodes"):
            state, _ = self.env.reset(
                seed=self.seed + episode)  # ADD epsiode so the seed is different for each episode

            rewards = []
            t = 0
            while t < episode_len:
                output = self.z.forward(torch.from_numpy(state / self.input_scale).float().to(self.device)).squeeze()

                if self.T_decay is None:
                    probs = torch.nn.functional.softmax(output / (self.T))
                else:
                    probs = torch.nn.functional.softmax(output / (self.T * self.T_decay ** episode))

                action = np.random.choice(self.actions, p=probs.detach().cpu().numpy())

                log_prob = torch.log(probs[action])

                # print("Action off device: ", action)
                new_state, reward, terminated, truncated, info = self.env.step(action)

                v = self.v.forward(torch.from_numpy(state / self.input_scale).float().to(self.device))
                new_v = 0
                if not terminated:
                    new_v = self.v.forward(torch.from_numpy(new_state / self.input_scale).float().to(self.device)).item()
                delta = new_v*self.gamma - v.item() + reward

                self.optimizer_v.zero_grad()
                v_loss = -delta*v
                v_loss.backward()
                self.optimizer_v.step()

                self.optimizer.zero_grad()
                loss = -delta * self.gamma**t * log_prob
                loss.backward()
                self.optimizer.step()

                rewards.append(reward)

                t += 1
                state = new_state
                if terminated or truncated:
                    break

            states, actions, log_probs, rewards = self.do_episode(episode_len, episode)

            total_reward = 0
            for i in range(len(rewards)):
                total_reward += self.gamma**i * rewards[i]
            total_rewards_v.append(total_reward)
            # print(total_reward)

        self.env.close()
        if plot_results:
            self.plot_reward(total_rewards_v)

        return total_rewards_v


if __name__ == "__main__":
    env_name = "Acrobot-v1"
    # env_name = "ALE/Assault-ram-v5"
    env = gym.make(env_name)
    model = ActorCritic(env, lr=0.003, lr_v=0.003, seed=25, T=4, input_scale=1)
    # model = Reinforce(env, lr=0.005, seed=25, T=8, T_decay=0.9975)
    model.train(100, 2000)

    new_env = gym.make(model.env.spec.id, render_mode='human')
    model.env.close()
    model.env = new_env
    model.do_episode(1000, 2000)
    model.do_episode(1000, 2000)
    model.do_episode(1000, 2000)
    model.do_episode(1000, 2000)
    model.do_episode(1000, 2000)

