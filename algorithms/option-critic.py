
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

class OptionCritic:
    def __init__(self, env: gym.Env, lr: float, lr_v: float, gamma: float = 1, T=0.1, T_decay=None, device='cpu', seed: int = 23, input_scale=1.0, num_options=4):
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
        self.num_options = num_options

        # We have 1d observations for this assignment, but adding more for more general case
        self.observation_size = 1
        for x in env.observation_space.shape:
            self.observation_size = self.observation_size * x

        self.actions = list(range(env.action_space.n))

        # Common backbone for all networks
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU())

        # From state to picking an action, deciding to stop an option
        self.option_policies = torch.nn.ModuleList()
        self.termination_policies = torch.nn.ModuleList()
        for i in range(num_options):
            self.option_policies.append(
                torch.nn.Sequential(
                    torch.nn.Linear(64, env.action_space.n)
                )
            )
            self.termination_policies.append(
                torch.nn.Sequential(
                    torch.nn.Linear(64, 1)
                )
            )
        self.option_policies.to(device)
        self.termination_policies.to(device)

        # From state to the value of each option
        self.Q_state_opts = torch.nn.Sequential(
            torch.nn.Linear(64, num_options)
        )
        self.Q_state_opts.to(device)

        self.reinit_weights()

        self.optimizer = torch.optim.RMSprop(self.z.parameters(), lr=self.lr)
        self.optimizer_v = torch.optim.RMSprop(self.v.parameters(), lr=self.lr_v)

    def seed_model(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def reinit_weights(self) -> None:
        for layer in self.Q_state_opts:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.uniform_(layer.weight, -0.001, 0.001)
                torch.nn.init.uniform_(layer.bias, -0.001, 0.001)

        for network in self.option_policies:
            for layer in network:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.uniform_(layer.weight, -0.001, 0.001)
                    torch.nn.init.uniform_(layer.bias, -0.001, 0.001)

        for network in self.termination_policies:
            for layer in network:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.uniform_(layer.weight, -0.001, 0.001)
                    torch.nn.init.uniform_(layer.bias, -0.001, 0.001)

    # Batch is a list of tuples from the replay buffer
    # All will be numpy arrays
    def update_Q(self, batch) -> None:
        # Batch is a list of tuples from the replay buffer
        # Update the Q function
        # Every element 0 of the tuple is the observation. We need to stack them to get a tensor of observations
        # Format is (observation, action, reward, observation_prime, terminated or truncated)
        batch = list(zip(*batch))
        states = torch.stack([torch.from_numpy(s).float() for s in batch[0]])
        actions = torch.tensor(batch[1])
        rewards = torch.tensor(batch[2])
        next_states = torch.stack([torch.from_numpy(s).float() for s in batch[3]])
        terminated = torch.tensor(batch[4], dtype=torch.bool)
        #print(terminated)

        # Get the next values
        with torch.no_grad():
            next_values = torch.max(self.Q_state_opts.forward(next_states), dim=1).values

        # Use terminated to mask next_values
        next_values[terminated] =0 
        y = rewards + self.gamma * next_values

        value_estimates = self.Q_state_opts.forward(states)
        state_value_estimates = value_estimates.gather(1, actions.unsqueeze(1)).squeeze(1)

        #print(state_value_estimates.shape)
        #print(y.shape)
        
        self.optimizer.zero_grad()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(state_value_estimates, y)
        #print("LOSS: ",loss)
        loss.backward()
        self.optimizer.step()
    
    # TODO: Implement according to g_t^1 of the paper
    def Q_swa(self):
        return None

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
            # TODO Choose ω according to an epsilon-soft policy over optionsπΩ(s)


            while t < episode_len:
                #TODO Choose action according to option policy
                

                #TODO δ ← r − QU (s, ω, a)
                
                #TODO if non terminal, add gamma * QU (s', ω, a)

                #TODO QU(swa) ← QU(swa) + α * δ


                #TODO Options improvements


                #TODO Critic improvements
                # I think just classic Q-learning is good here



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
    env_name = "ALE/Assault-ram-v5"
    # env_name = "ALE/Assault-ram-v5"
    env = gym.make(env_name)
    model = OptionCritic(env, lr=0.003, lr_v=0.003, seed=25, T=4, input_scale=1)
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

