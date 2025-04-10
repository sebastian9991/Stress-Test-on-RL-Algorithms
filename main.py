from unittest import result
import gymnasium as gym
from algorithms.option_critic import OptionCritic
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Save results dict as a json file
import json
from itertools import product
from plot_runs import plot_runs, plot_rewards_over_time, plot_rewards_over_time_files


VALID_MODELS = ['OptionCritic']

option_critic_hyperparams = {
    "lr": [ 0.001, 0.0001],
    "gamma": [0.1, 0.01],
    "T": [2],
    "num_options": [2, 4],
    "epsilon": [0.1, 0.2],
    "overall_alpha": [0.0001, 0.001],
}

def run_hyperparam_search(model:str, hyperparams: dict, independent_trials: int = 3, num_episodes: int = 1000, envs: list = ['Acrobot-v1', 'ALE/Assault-ram-v5'] ):
    
    # Seeds, 10 seeds for max 10 runs
    seeds = [563133, 665248, 414684, 18048, 863607, 984126, 969211, 736317, 
    406635, 514876]

    if model not in VALID_MODELS:
        raise ValueError(f"Model must be in {VALID_MODELS}, but got {model}. Add {model} to the function.")
    if independent_trials > len(seeds):
        raise ValueError(f"Independent trials cannot be greater than {len(seeds)}. Add more random seeds to the \"seeds\" list.")

    for env_name in envs:
        results = {}
        for combo in product(*hyperparams.values()):
            # Zip keys and values into a dictionary for clarity
            config = dict(zip(hyperparams.keys(), combo))
            rewards = np.zeros((independent_trials, num_episodes))
            print(config)

            for i in tqdm(range(independent_trials), desc=f"Independent runs for {env_name}", leave = False):
                env = gym.make(env_name)              

                if model == 'OptionCritic':
                    agent = OptionCritic(env = env, **config, seed=seeds[i])
                
                rewards[i] = agent.train(num_episodes, 1e10)

            result_name = ""
            for key in config.keys():
                result_name += f"{key}={config[key]},"
            results[result_name] = rewards

            # Convert numpy arrays to lists for JSON serialization
            # Do this after every combination of hyperparameters to avoid losing data
            results_serializable = {k: v.tolist() for k, v in results.items()}
            env_name_valid = env_name.replace('/', '_')
            with open(f'{model}_{env_name_valid}_results.json', 'w') as f:
                json.dump(results_serializable, f)

def main():
    print("Main Experiement ran from here:")

    run_hyperparam_search(
        model='OptionCritic',
        hyperparams=option_critic_hyperparams,
        independent_trials=3,
        num_episodes=10,
        envs=['CartPole-v1']
    )
    
    plot_runs(files = ['OptionCritic_CartPole-v1_results.json'], hyperparams = option_critic_hyperparams, show = True)
    


if __name__ == "__main__":
    main()
