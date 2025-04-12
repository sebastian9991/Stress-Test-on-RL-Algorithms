import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from policies.policy import Policy
from utils.helper_gradient import (conjugate_gradient, flat_grad,
                                   kl_div_categorical)


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

        self.policy_optimizer = optim.Adam(self.policy.policy_net.paramaters())

    def seed_model(self, seed: int) -> None:
        np.random.seed(seed)
        self.seed = seed

    def run_episode(self, max_episode_length):
        states, actions, rewards = [], [], []
        state = self.env.reset()  # We let s_0 approx p_0 be defined by the env
        if isinstance(state, tuple):
            state = state[0]

        eps = 0
        while eps < max_episode_length:
            action_pair = self.policy.select_action(state)
            action = action_pair[0]
            next_state, reward, done, _ = self.env.step(action)

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
        q_values = np.zeros((self.state_dim, self.action_dim))
        for t in reversed(range(len(states))):
            G = rewards[t] + gamma * G
            state = states[t]
            action = actions[t]
            q_values[state, action] = G
        return q_values

    def surrogate_loss_state(self, state, q_values):
        state = torch.tensor(state, dtype=torch.float32).unqueeze(0)
        action_probs = self.policy.policy_net(state)
        action_probs = action_probs.squeeze(0)

        q_values_for_state = torch.tensor(q_values[int(state)], dtype=torch.float32)

        loss = torch.sum(action_probs * q_values_for_state)
        return loss

    def surrogate_loss(self, states, q_values):
        total_loss = 0
        for state in states:
            total_loss += self.surrogate_loss_state(state, q_values)
        return total_loss / len(states)

    def update_policy(self, step) -> None:
        self.policy_optimizer.zero_grad()
        step.backward()
        self.policy_optimizer.step()

    def update_agent(self, states, action, rewards, q_values, delta=0.01):

        L = self.surrogate_loss(states, q_values)
        probabilites = self.policy.get_probabilites_states(states)
        KL = kl_div_categorical(probabilites, probabilites)

        paramaters = list(self.policy.policy_net.paramaters())

        g = flat_grad(L, paramaters, retain_graph=True)
        d_kl = flat_grad(KL, paramaters, create_graph=True)

        # Hessian vector product on KL
        def Hessian_vector_product(v):
            return flat_grad(d_kl @ v, paramaters, retain_graph=True)

        search_dir = conjugate_gradient(Hessian_vector_product, g)
        maximal_step_length = torch.sqrt(
            2 * delta / (search_dir @ Hessian_vector_product(search_dir))
        )
        max_step = maximal_step_length * search_dir

        def line_search(step):
            self.update_policy(step)

            with torch.no_grad():
                L_new = self.surrogate_loss(states, q_values)
                probabilites_new = self.policy.get_probabilities_states(states)
                KL_new = kl_div_categorical(probabilites_new, probabilites)

            L_improvement = L_new - L

            if L_improvement > 0 and KL_new <= delta:
                return True

            self.update_policy(-step)
            return False

        i = 1
        while not line_search((0.9**i) * max_step) and i < 10:
            i += 1

    def train(self, max_iterations=1000, max_episode_length=1000):

        for _ in range(0, max_iterations):
            states, actions, rewards = self.run_episode(max_episode_length)
            q_values = self.compute_q_values_single_path(states, actions, rewards)
            self.update_agent(states, actions, rewards, q_values)
