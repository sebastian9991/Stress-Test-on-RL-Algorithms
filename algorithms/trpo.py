import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from policies.policy import Policy

######
# TRPO with VINE and with single_path as paramaters
# Step-by-Step
# 1. Use the single_path or vine procedures to collect a set of state-action pairs along with the Monte-Carlo estimates of their Q-Values.
# 2. By Averaging over the samples, construct the estimated objective L(theta)
# 3. Solve the constrain optimization problem approzimately through the conjugate gradient theorem and a line search. Which you can use to get the update for the policy's paramater vector theta.
# a.)
#####


class TRPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        delta: float,
        policy: Policy,
        env: gym.Env,
        seed: int = 23,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.delta = delta
        self.env = env
        self.policy = policy
        self.seed = seed

        self.q_values = np.zeros(
            (self.state_dim, self.action_dim)
        )  # Initalize to zero. Is this correct?

    def seed_model(self, seed: int) -> None:
        np.random.seed(seed)
        self.seed = seed

    def run_episode(self):
        states, actions, rewards = [], [], []
        state = self.env.reset()  # We let s_0 approx p_0 be defined by the env
        if isinstance(state, tuple):
            state = state[0]

        while True:
            action_pair = self.policy.select_action(state)
            action = action_pair[0]
            next_state, reward, done, _ = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

            if done:
                break

        return states, actions, rewards

    def compute_q_values_single_path(self, states, actions, rewards, gamma=0.99):
        G = 0
        for t in reversed(range(len(states))):
            G = rewards[t] + gamma * G
            state = states[t]
            action = actions[t]
            self.q_values[state, action] = G

        



    # TODO: Add these as paramaters to init
    def train(self, max_iterations=1000, max_episode_length=1000):
        optimizer = optim.Adam(self.policy.policy_net.paramaters())
