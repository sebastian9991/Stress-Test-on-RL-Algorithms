
from collections import deque
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm


class DQN:
    def __init__(self, env: gym.Env, lr: float, gamma: float = 0.99, epsilon = 0.1, device = 'cpu', seed: int = 23):
        self.env = env
        self.lr = lr
        self.seed = seed
        self.device = device
        self.seed_model(seed)
        self.gamma = gamma
        self.epsilon = epsilon

        # We have 1d observations for this assignment, but adding more for more general case
        self.observation_size = 1
        for x in env.observation_space.shape:
            self.observation_size = self.observation_size * x
        
        self.actions = list(range(env.action_space.n))
        self.Q = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, env.action_space.n)
        )

        self.reinit_weights()
        
        self.Q.to(device)
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=lr)
    
    # Batch is a list of tuples from the replay buffer
    # All will be numpy arrays
    def update(self, batch) -> None:
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
            next_values = torch.max(self.Q.forward(next_states), dim=1).values

        # Use terminated to mask next_values
        next_values[terminated] =0 
        y = rewards + self.gamma * next_values

        value_estimates = self.Q.forward(states)
        state_value_estimates = value_estimates.gather(1, actions.unsqueeze(1)).squeeze(1)

        #print(state_value_estimates.shape)
        #print(y.shape)
        
        self.optimizer.zero_grad()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(state_value_estimates, y)
        #print("LOSS: ",loss)
        loss.backward()
        self.optimizer.step()

        
    def seed_model(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def reinit_weights(self) -> None:
        for layer in self.Q:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.uniform_(layer.weight, -0.001, 0.001)
                torch.nn.init.uniform_(layer.bias, -0.001, 0.001)

    def select_action(self, observation: torch.Tensor) -> int:
        # Take in observation from env, return action
        rand = np.random.rand() 
        if rand < self.epsilon:
            return np.random.randint(0,self.env.action_space.n)
        else:
            with torch.no_grad():
                estimates = self.Q.forward(torch.from_numpy(observation).float().to(self.device))
            return torch.argmax(estimates).item()        

    def train(self, number_of_episodes: int, max_iterations: int = 1000, render = True, use_buffer = True, 
              replay: int = 1000000, batch_size: int = 16,stress_config: dict = None) -> None:
        # Collect episode 
        # update replay buffer if you have one
        # update the Neural network 
        # Replay and batch size only used if use_buffer is True

        self.seed_model(self.seed)

        D = deque(maxlen=replay)
        rewards = []

        for episode in tqdm(range(number_of_episodes), leave=False, desc="Episodes"):
            if stress_config is not None and (episode == 500):
                observation, _ = self.env.reset(seed=self.seed + episode, **stress_config)
                print("STRESS TEST BEGUN")
            else:
                observation, _ = self.env.reset(seed=self.seed + episode)
            episode_rewards = []
            #print(type(observation))
            #print(observation.shape)
            t = -1
            while t < max_iterations:
                t += 1
                action = self.select_action(observation)
                
                #print("Action off device: ", action)
                observation_prime, reward, terminated, truncated, info = self.env.step(action)                    

                episode_rewards.append(reward)
                
                # Replay buffer
                if use_buffer:
                    D.append((observation, action, reward, observation_prime, terminated or truncated))
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
                    batch = [(observation, action, reward, observation_prime, terminated or truncated)]
                    
                self.update(batch)
                
                observation = observation_prime
                if terminated or truncated:
                    break
            rewards.append(sum(episode_rewards))

        self.env.close()

        return rewards


                  