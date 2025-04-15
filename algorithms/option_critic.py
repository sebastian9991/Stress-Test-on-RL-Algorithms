import math
from pdb import run

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.rmsprop
from torch import nn, softmax
from tqdm import tqdm

gym.register_envs(ale_py)


class OptionCritic:
    def __init__(
        self,
        env: gym.Env,
        lr: float = 0.001,
        gamma: float = 0.99,
        T=2,
        T_decay=None,
        device="cpu",
        seed: int = 23,
        input_scale=1.0,
        num_options=2,
        epsilon=0.1,
        alpha_critic=0.0001,
        alpha_option=0.001,
        alpha_termination=0.0001,
        overall_alpha=None,
    ):
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

        # Set all alphas to same value
        if overall_alpha is not None:
            self.alpha_option = overall_alpha
            self.alpha_termination = overall_alpha
            self.alpha_critic = overall_alpha

        # We have 1d observations for this assignment, but adding more for more general case
        self.observation_size = 1
        for x in env.observation_space.shape:
            self.observation_size = self.observation_size * x

        self.actions = list(range(env.action_space.n))

        # Common backbone for all networks
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, 256),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
        )

        # From state to picking an action, deciding to stop an option
        self.option_policies = torch.nn.ModuleList()
        self.termination_policies = torch.nn.ModuleList()
        for i in range(num_options):
            self.option_policies.append(
                torch.nn.Sequential(
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, env.action_space.n),
                )
            )
            self.termination_policies.append(
                torch.nn.Sequential(
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 1),
                    torch.nn.Sigmoid(),
                )
            )
        self.option_policies.to(device)
        self.termination_policies.to(device)

        # From state to the value of each option
        self.Q_state_opts = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, num_options)
        )
        self.Q_state_opts.to(device)

        self.parameters_to_optimize = list(self.backbone.parameters()) + list(
            self.Q_state_opts.parameters()
        )
        for i in range(num_options):
            self.parameters_to_optimize += list(self.option_policies[i].parameters())
            self.parameters_to_optimize += list(
                self.termination_policies[i].parameters()
            )

        self.optimizer = torch.optim.AdamW(self.parameters_to_optimize, lr=self.lr)

        self.reinit_weights()

    def seed_model(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def reinit_weights(self) -> None:
        for layer in self.Q_state_opts:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

        for network in self.option_policies:
            for layer in network:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

        for network in self.termination_policies:
            for layer in network:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

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
        # print(terminated)

        # Get the next values
        with torch.no_grad():
            state_encoding = self.backbone.forward(next_states / self.input_scale)
            next_values = torch.max(
                self.Q_state_opts.forward(state_encoding), dim=1
            ).values

        # Use terminated to mask next_values
        next_values[terminated] = 0
        y = rewards + self.gamma * next_values

        state_encoding = self.backbone.forward(states / self.input_scale)
        value_estimates = self.Q_state_opts.forward(state_encoding)
        # print("Value estimates: ", value_estimates)
        # print("Actions: ", actions)
        state_value_estimates = value_estimates.gather(1, actions.unsqueeze(1)).squeeze(
            1
        )

        # print(state_value_estimates.shape)
        # print(y.shape)

        self.optimizer.zero_grad()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(state_value_estimates, y)
        # print("LOSS: ",loss)
        loss.backward()
        self.optimizer.step()

    # TODO: Implement according to g_t^1 of the paper
    def Q_swa(self, reward, state_encoding, option):
        # Should implement r + γ(1 − βω,θ(s′))QΩ(s′, ω) +γβω,θ(s′) max ω QΩ(s′,  ̄ω)
        Qw = self.Q_state_opts.forward(state_encoding)
        Qw_this = Qw[option]
        Qw_best = Qw.max()

        termination_prob = self.termination_policies[option].forward(state_encoding)

        result = (
            reward
            + self.gamma * (1 - termination_prob) * Qw_this
            + self.gamma * termination_prob * Qw_best
        )
        return result

    def epsilon_soft_policy(self, action_logits):
        """
        Epsilon soft policy. Returns a distribution over actions
        Args:
            action_logits: logits of the actions
        Returns:
            action_probs: epsilon soft distribution over actions
        """
        action_probs = torch.nn.functional.softmax(action_logits / self.T, dim=0)
        # print("Action probs after softmax, before softening: ", action_probs)
        # Make epsilon soft
        soft_probs = (1 - self.epsilon) * action_probs + (
            self.epsilon / len(action_logits)
        )
        return soft_probs

    def do_episode(
        self,
        number_of_episodes: int,
        max_iterations: int,
        plot_results=False,
        use_buffer=True,
        replay=1000000,
        batch_size=16,
    ):
        # Collect episode
        # update replay buffer if you have one
        # update the Neural network
        # Replay and batch size only used if use_buffer is True

        self.seed_model(self.seed)
        total_rewards_v = []

        for episode in tqdm(range(number_of_episodes), leave=False, desc="Episodes"):
            state, _ = self.env.reset(
                seed=self.seed + episode
            )  # ADD epsiode so the seed is different for each episode

            rewards = []
            t = 0

            state_encoding = self.backbone.forward(
                torch.from_numpy(state / self.input_scale).float().to(self.device)
            )

            # TODO Choose ω according to an epsilon-soft policy over optionsπΩ(s)
            with torch.no_grad():
                action_probs = self.Q_state_opts.forward(state_encoding)
                best_action = torch.argmax(action_probs).item()
                opts = np.ones(self.num_options) * self.epsilon / self.num_options
                opts[best_action] = 1 - self.epsilon + self.epsilon / self.num_options
                option = np.random.choice(self.num_options, p=opts)

            episode_reward = 0

            while t < max_iterations:

                ### TAKE ACTION ###
                # Load the common backbone to get our state encoding
                state_encoding = self.backbone.forward(
                    torch.from_numpy(state / self.input_scale).float().to(self.device)
                )

                # Choose action according to option policy
                action_logits = self.option_policies[option].forward(state_encoding)
                # print("Action probs before softmax: ", action_logits)
                action_probs = self.epsilon_soft_policy(action_logits)
                # print("Action probs after softmax: ", action_probs)
                action = np.random.choice(
                    self.actions, p=action_probs.detach().cpu().numpy()
                )
                observation, reward, terminated, truncated, info = self.env.step(action)

                termination_prob = self.termination_policies[option].forward(
                    state_encoding
                )
                new_state_encoding = self.backbone.forward(
                    torch.from_numpy(observation / self.input_scale)
                    .float()
                    .to(self.device)
                )

                ### CHANGE OPTION ###
                if np.random.rand() < termination_prob.item():
                    with torch.no_grad():
                        new_option_values = self.Q_state_opts.forward(
                            new_state_encoding
                        )
                        # print("New option values: ", new_option_values)
                        option_probs = self.epsilon_soft_policy(new_option_values)
                        # print("Option probs: ", option_probs)
                        option = np.random.choice(
                            self.num_options, p=option_probs.detach().cpu().numpy()
                        )
                    # print("New option: ", option)

                # print("action: ", action)

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

    def train(
        self,
        number_of_episodes: int,
        max_iterations: int,
        plot_results=False,
        use_buffer=True,
        replay=1000000,
        batch_size=16,
    ):
        # Collect episode
        # update replay buffer if you have one
        # update the Neural network
        # Replay and batch size only used if use_buffer is True

        self.seed_model(self.seed)
        total_rewards_v = []

        for episode in tqdm(range(number_of_episodes), leave=False, desc="Episodes"):
            state, _ = self.env.reset(
                seed=self.seed + episode
            )  # ADD epsiode so the seed is different for each episode

            rewards = []
            t = 0

            state_encoding = self.backbone.forward(
                torch.from_numpy(state / self.input_scale).float().to(self.device)
            )

            # TODO Choose ω according to an epsilon-soft policy over optionsπΩ(s)
            with torch.no_grad():
                action_probs = self.Q_state_opts.forward(state_encoding)
                best_action = torch.argmax(action_probs).item()
                opts = np.ones(self.num_options) * self.epsilon / self.num_options
                opts[best_action] = 1 - self.epsilon + self.epsilon / self.num_options
                option = np.random.choice(self.num_options, p=opts)

            episode_reward = 0

            while t < max_iterations:

                ### TAKE ACTION ###
                # Load the common backbone to get our state encoding
                state_encoding = self.backbone.forward(
                    torch.from_numpy(state / self.input_scale).float().to(self.device)
                )

                # Choose action according to option policy
                action_logits = self.option_policies[option].forward(state_encoding)
                # print("Action probs before softmax: ", action_logits)
                action_probs = self.epsilon_soft_policy(action_logits)
                # print("Action probs after softmax: ", action_probs)
                action = np.random.choice(
                    self.actions, p=action_probs.detach().cpu().numpy()
                )
                observation, reward, terminated, truncated, info = self.env.step(action)

                ### UPDATE THE Q FUNCTION ###

                # Create the target for the Q function
                with torch.no_grad():
                    next_state_encoding = self.backbone.forward(
                        torch.from_numpy(observation / self.input_scale)
                        .float()
                        .to(self.device)
                    )
                    next_value = self.Q_state_opts.forward(next_state_encoding)
                    next_state_option_value = next_value[option]
                    best_action = torch.argmax(next_value).item()

                    next_state_termination_prob = self.termination_policies[
                        option
                    ].forward(next_state_encoding)

                    next_state_value = (
                        (1 - next_state_termination_prob) * next_state_option_value
                        + next_state_termination_prob * next_value[best_action]
                    )
                    if not terminated and not truncated:
                        current_state_value_target = (
                            reward + self.gamma * next_state_value
                        )
                        current_state_value_target = current_state_value_target
                    else:
                        current_state_value_target = reward
                    current_state_value_target = torch.tensor(
                        current_state_value_target
                    )
                    # print("Current state value target: ", current_state_value_target, "Estimate", self.Q_state_opts.forward(state_encoding)[option])

                # Update the Q function
                self.optimizer.zero_grad()
                # print(current_state_value_target)
                # print(self.Q_state_opts.forward(state_encoding)[option])
                q_loss = torch.nn.MSELoss()(
                    self.Q_state_opts.forward(state_encoding)[option],
                    current_state_value_target,
                )
                # print("Q LOSS: ", q_loss)
                q_loss.backward()
                self.optimizer.step()

                ### UPDATE THE OPTION POLICY ###
                # print(action_probs)
                # Recompute the action probs for a fresh graph
                state_encoding = self.backbone.forward(
                    torch.from_numpy(state / self.input_scale).float().to(self.device)
                )
                action_logits = self.option_policies[option].forward(state_encoding)
                action_probs = self.epsilon_soft_policy(action_logits)

                # Get a baseline for the advantage
                with torch.no_grad():
                    baseline = torch.sum(
                        self.Q_state_opts(state_encoding)
                        * self.epsilon_soft_policy(self.Q_state_opts(state_encoding))
                    )
                    advantage = current_state_value_target - baseline
                    option_policy_advantage = advantage.item()
                    # print("Option advantage", option_policy_advantage, baseline, current_state_value_target)

                log_prob = torch.log(action_probs[action] + 1e-6)
                # print("Log prob: ", log_prob)
                self.optimizer.zero_grad()
                option_loss = -self.alpha_option * advantage * log_prob
                # print("OPTION LOSS: ", option_loss)
                option_loss.backward()
                self.optimizer.step()

                ### UPDATE THE TERMINATION POLICY ###

                # Recompute the action probs for a fresh graph
                new_state_encoding = self.backbone.forward(
                    torch.from_numpy(state / self.input_scale).float().to(self.device)
                )
                termination_prob = self.termination_policies[option].forward(
                    new_state_encoding
                )

                with torch.no_grad():
                    new_option_values = self.Q_state_opts.forward(new_state_encoding)
                    current_option_value = new_option_values[option]
                    option_probs = self.epsilon_soft_policy(new_option_values)

                    new_value = torch.sum(new_option_values * option_probs)

                    advantage = (current_option_value - new_value).item()
                    # print("Termination advantage", advantage)

                self.optimizer.zero_grad()
                termination_loss = (
                    -self.alpha_termination * advantage * termination_prob
                )  # TODO IS THIS CORRECT??? +/-?
                # print("Loss: ", termination_loss, " Advantage: ", advantage, " Termination prob: ", termination_prob)
                # print("TERMINATION LOSS: ", termination_loss)
                termination_loss.backward()
                # Print computation graph
                self.optimizer.step()

                ### CHANGE OPTION ###
                if np.random.rand() < termination_prob.item():
                    with torch.no_grad():
                        new_option_values = self.Q_state_opts.forward(
                            new_state_encoding
                        )
                        # print("New option values: ", new_option_values)
                        option_probs = self.epsilon_soft_policy(new_option_values)
                        # print("Option probs: ", option_probs)
                        option = np.random.choice(
                            self.num_options, p=option_probs.detach().cpu().numpy()
                        )
                    # print("New option: ", option)

                # print("action: ", action)

                episode_reward += reward
                rewards.append(reward)

                t += 1
                state = observation
                if terminated or truncated:
                    break
            # if episode % 10 == 0:
            #     print(
            #         "Action probs: ",
            #         action_logits,
            #         " Option Probs: ",
            #         new_option_values,
            #         " Mean reward last 10: ",
            #         np.mean(total_rewards_v[-10:]),
            #         " Advantages(opt, term): ",
            #         option_policy_advantage,
            #         advantage,
            #         "State value: ",
            #         current_state_value_target,
            #     )

            # print("Episode: ", episode)
            # print("Episode reward: ", episode_reward)
            # print("Episode length: ", t)
            total_rewards_v.append(episode_reward)

        self.env.close()

        return total_rewards_v


def run_hyperparam_search(
    env: gym.Env, number_of_episodes: int, max_iterations: int, plot_results=False
):
    # Hyperparameter search
    results = {}
    for lr in [0.1, 0.01, 0.001]:
        for alpha in [0.1, 0.01, 0.001]:
            for num_options in [2, 4, 8]:
                model = OptionCritic(
                    env,
                    lr=lr,
                    alpha_critic=alpha,
                    alpha_option=alpha,
                    alpha_termination=alpha,
                    num_options=num_options,
                )
                temp_results = model.train(number_of_episodes, max_iterations)
                if plot_results:
                    plt.plot(range(max_iterations), temp_results)
                    plt.title(f"lr: {lr}, alpha: {alpha}, num_options: {num_options}")
                    plt.show()
                results[f"lr{lr}_a{alpha}_opt{num_options}"] = temp_results
    return results


if __name__ == "__main__":
    env_name = "CartPole-v1"
    # env_name = "Acrobot-v1"
    # env_name = "ALE/Assault-ram-v5"
    env = gym.make(env_name)

    # results = run_hyperparam_search(env, 100, 2000000, plot_results=True)
    # save results to json
    # import json
    # with open("results.json", "w") as f:
    # json.dump(results, f)
    model = OptionCritic(
        env,
        seed=25,
        T=4,
        input_scale=1,
        num_options=2,
        lr=0.00005,
        alpha_critic=0.001,
        alpha_option=0.1,
        alpha_termination=0.1,
    )
    # model = Reinforce(env, lr=0.005, seed=25, T=8, T_decay=0.9975)
    num_epsides = 2000
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
    model.do_episode(1, 2000000)
