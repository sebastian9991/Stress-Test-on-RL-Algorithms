
from collections import deque
from re import S
import re
import stat
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch import nn, softmax
import torch.optim.rmsprop
from tqdm import tqdm
import math
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

class OptionCritic:
    def __init__(self, env: gym.Env, lr: float = 0.001, gamma: float = 1, T=0.1, T_decay=None, device='cpu', 
                 seed: int = 23, input_scale=1.0, num_options=4, epsilon=0.001, alpha_critic =0.0001, alpha_option=0.00001,
                 alpha_termination=0.1):
        self.env = env
        self.seed = seed
        self.device = device
        self.lr = lr
        self.seed_model(seed)
        self.gamma = gamma  # TODO
        self.T = T
        self.T_decay = T_decay
        self.input_scale = input_scale
        self.num_options = num_options
        self.epsilon = epsilon
        self.alpha_critic = alpha_critic
        self.alpha_option = alpha_option
        self.alpha_termination = alpha_termination

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
                    torch.nn.Linear(64, 1),
                    torch.nn.Sigmoid()
                )
            )
        self.option_policies.to(device)
        self.termination_policies.to(device)

        # From state to the value of each option
        self.Q_state_opts = torch.nn.Sequential(
            torch.nn.Linear(64, num_options)
        )
        self.Q_state_opts.to(device)

        self.parameters_to_optimize = list(self.backbone.parameters()) + list(self.Q_state_opts.parameters())
        for i in range(num_options):
            self.parameters_to_optimize += list(self.option_policies[i].parameters())
            self.parameters_to_optimize += list(self.termination_policies[i].parameters())

        self.optimizer = torch.optim.Adam(self.parameters_to_optimize, lr=self.lr)

        self.reinit_weights()

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
            state_encoding = self.backbone.forward(next_states / self.input_scale)
            next_values = torch.max(self.Q_state_opts.forward(state_encoding), dim=1).values

        # Use terminated to mask next_values
        next_values[terminated] =0 
        y = rewards + self.gamma * next_values

        state_encoding = self.backbone.forward(states / self.input_scale)
        value_estimates = self.Q_state_opts.forward(state_encoding)
        #print("Value estimates: ", value_estimates)
        #print("Actions: ", actions)
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
    def Q_swa(self, reward, state_encoding, option):

        
            Qw = self.Q_state_opts.forward(state_encoding)
            Qw_this = Qw[option]
            Qw_best = Qw.max()

            termination_prob = self.termination_policies[option].forward(state_encoding)

            result = reward + self.gamma*(1 - termination_prob) * Qw_this + self.gamma * termination_prob * Qw_best
            return result

    def train(self, num_episodes: int, episode_len: int, plot_results=False, use_buffer = True, replay =1000000, batch_size = 16):
        # Collect episode
        # update replay buffer if you have one
        # update the Neural network
        # Replay and batch size only used if use_buffer is True

        self.seed_model(self.seed)
        total_rewards_v = []
        D = deque(maxlen=replay)

        for episode in tqdm(range(num_episodes), leave=False, desc="Episodes"):
            state, _ = self.env.reset(
                seed=self.seed + episode)  # ADD epsiode so the seed is different for each episode

            rewards = []
            t = 0

            state_encoding = self.backbone.forward(torch.from_numpy(state / self.input_scale).float().to(self.device))

            # TODO Choose ω according to an epsilon-soft policy over optionsπΩ(s)
            with torch.no_grad():
                action_probs = self.Q_state_opts.forward(state_encoding)
                best_action = torch.argmax(action_probs).item()
                opts = np.ones(self.num_options) * self.epsilon / self.num_options
                opts[best_action] = 1 - self.epsilon + self.epsilon / self.num_options
                option = np.random.choice(self.num_options, p=opts)

            episode_reward = 0


            while t < episode_len:
                # Load the common backbone to get our state encoding
                state_encoding = self.backbone.forward(torch.from_numpy(state / self.input_scale).float().to(self.device))

                # Choose action according to option policy
                action_probs = self.option_policies[option].forward(state_encoding)
                #print("Action probs before softmax: ", action_probs)
                action_probs = torch.nn.functional.softmax(action_probs, dim=0)
                action_probs = (1 - self.epsilon) * action_probs + (self.epsilon / len(self.actions)) 
                #print("Action probs: ", action_probs)

                action = np.random.choice(self.actions, p=action_probs.detach().cpu().numpy())

                observation, reward, terminated, truncated, info = self.env.step(action)

                #TODO δ ← r − QU (s, ω, a)
                QU_swa = self.Q_swa(reward, state_encoding, option)
                delta = reward - QU_swa

                #TODO if non terminal, add gamma * QU (s', ω, a)
                with torch.no_grad():
                        new_state_encoding = self.backbone.forward(torch.from_numpy(observation / self.input_scale).float().to(self.device))
                if not terminated and not truncated:
                    delta = delta + self.Q_swa(0, new_state_encoding, option) # Reward of 0 for Q_swa as per the pseudocode from paper


                #TODO QU(swa) ← QU(swa) + α * δ
                QU_swa = QU_swa + self.alpha_critic * delta


                #TODO Options improvements
                log_prob = torch.log(action_probs[action])
                self.optimizer.zero_grad()
                option_loss = - self.alpha_option * QU_swa * log_prob
                #print("OPTION LOSS: ", option_loss)
                option_loss.backward(retain_graph=True)
                self.optimizer.step()

                #TODO Termination improvements
                state_encoding = self.backbone.forward(torch.from_numpy(state / self.input_scale).float().to(self.device))
                termination_prob = self.termination_policies[option].forward(state_encoding)

                with torch.no_grad():
                    new_value = self.Q_state_opts.forward(new_state_encoding)

                    best_action = torch.argmax(new_value).item()
                    opts = np.ones(self.num_options) * self.epsilon / self.num_options
                    opts[best_action] = 1 - self.epsilon + self.epsilon / self.num_options

                    value = torch.sum(new_value * opts)
                    advantage = value - new_value[option]
                    advantage = advantage.item()

                termination_loss = - self.alpha_termination * advantage * torch.log(termination_prob + 1e-6)
                self.optimizer.zero_grad()
                #print("TERMINATION LOSS: ", termination_loss)
                termination_loss.backward()
                self.optimizer.step()

                
                #TODO Critic improvements
                # I think just classic Q-learning is good here
                # Replay buffer
                if replay:
                    D.append((state, option, reward, observation, terminated or truncated))
                    if len(D) > replay:
                        #D.pop(0)
                        pass
                    if len(D) > batch_size:
                        try:
                            # I THINK IT's looking across dimensions. Look later
                            batch_indexes = np.random.choice(len(D), batch_size)
                            batch = [D[i] for i in batch_indexes]
                        except:
                            print("TIME: ",t)
                            print(D[0])
                            exit()
                    else:
                        batch = D
                # No replay buffer
                else:
                    batch = [(state, option, reward, observation, terminated or truncated)]
                #print("Batch: ", len(batch))
                self.update_Q(batch)
                

                #TODO If option policy terminates, choose new option
                if np.random.rand() < termination_prob.item():
                    with torch.no_grad():
                        option_probs = self.Q_state_opts.forward(new_state_encoding)
                        best_action = torch.argmax(option_probs).item()
                        opts = np.ones(self.num_options) * self.epsilon / self.num_options
                        opts[best_action] = 1 - self.epsilon + self.epsilon / self.num_options
                        option = np.random.choice(self.num_options, p=opts)
                    #print("New option: ", option)

                #print("action: ", action)




                episode_reward += reward
                rewards.append(reward)

                t += 1
                state = observation
                if terminated or truncated:
                    break
            print("Episode: ", episode)
            print("Episode reward: ", episode_reward)
            print("Episode length: ", t)

        self.env.close()

        return total_rewards_v


if __name__ == "__main__":
    env_name = "Acrobot-v1"
    # env_name = "ALE/Assault-ram-v5"
    env = gym.make(env_name)
    model = OptionCritic(env, seed=25, T=4, input_scale=1, num_options=4, epsilon=0.2)
    # model = Reinforce(env, lr=0.005, seed=25, T=8, T_decay=0.9975)
    model.train(100, 1000)

    new_env = gym.make(model.env.spec.id, render_mode='human')
    model.env.close()
    model.env = new_env
    model.do_episode(1000, 2000)
    model.do_episode(1000, 2000)
    model.do_episode(1000, 2000)
    model.do_episode(1000, 2000)
    model.do_episode(1000, 2000)

