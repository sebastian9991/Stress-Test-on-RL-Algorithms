from typing import Dict

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

gym.register_envs(ale_py)


class RandomAgent:
    def __init__(self, env: gym.Env, do_stress_test: bool = True, seed: int = 23):
        self.env: gym.Env = env
        self.seed = seed
        self.do_stress_test = do_stress_test
        self.actions = range(env.action_space.n)

        # We have 1d observations for this assignment, but adding more for more general case
        self.observation_size = 1
        for x in env.observation_space.shape:
            self.observation_size = self.observation_size * x

        self.seed_model(seed)

    def seed_model(self, seed):
        # Set random seeds for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Initialize environment with seed
        if seed is not None:
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)

    def train(
        self,
        stress_config: Dict,
        number_of_episodes: int = 1000,
        max_iterations: int = 1000,
    ):
        # Collect episode
        # update replay buffer if you have one
        # update the Neural network
        # Replay and batch size only used if use_buffer is True

        self.seed_model(self.seed)
        total_rewards_v = []

        for episode in tqdm(range(number_of_episodes), leave=False, desc="Episodes"):
            if self.do_stress_test and (episode == 100 or episode == 500):
                self.env.stress_test(**stress_config)
            state, _ = self.env.reset(
                seed=self.seed + episode
            )  # ADD epsiode so the seed is different for each episode
            episode_reward = 0
            t = 0
            while t < max_iterations:

                action = np.random.choice(self.actions)
                observation, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward

                t += 1
                if terminated or truncated:
                    break
            # print("Episode: ", episode)
            # print("Episode reward: ", episode_reward)
            # print("Episode length: ", t)
            total_rewards_v.append(episode_reward)

        self.env.close()

        return total_rewards_v


# For testing purposes
if __name__ == "__main__":
    env_name = "CartPole-v1"
    #env_name = "ALE/Assault-ram-v5"
    env_name = "ALE/Pacman-ram-v5"
    env = gym.make(env_name)
    model = RandomAgent(env, seed=25)
    # model = Reinforce(env, lr=0.005, seed=25, T=8, T_decay=0.9975)
    num_epsides = 800
    results = model.train(num_epsides, 2000000)

    # plot results
    plt.plot(range(num_epsides), results)
    plt.show()

    smoothed_results = []
    window = 25
    smoothed_results = [
        np.mean(results[max(0, i - window) : min(len(results) - 1, i + window)])
        for i in range(len(results))
    ]
    plt.plot(range(num_epsides), smoothed_results)
    plt.title("Smoothed results")
    plt.show()

    new_env = gym.make(model.env.spec.id, render_mode="human")
    model.env.close()
    model.env = new_env
    model.train(1, 2000000)
