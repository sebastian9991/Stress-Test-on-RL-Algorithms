import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from policies.policy import Policy


class TRPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        policy: Policy,
        env: gym.Env,
        seed: int = 23,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env
        self.policy = policy
        self.seed = seed

    def seed_model(self, seed: int) -> None:
        np.random.seed(seed)
        self.seed = seed
