# Save results dict as a json file
import json
import os
from itertools import product

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from ale_py import env
from tqdm import tqdm

from algorithms.actor_critic import ActorCritic
from algorithms.dqn import DQN
from algorithms.option_critic import OptionCritic
from algorithms.random_agent import RandomAgent
from algorithms.trpo import TRPO
from mutable_ale.mutable_ALE import MutableAtariEnv
from mutable_ale.mutable_cartpole import MutableCartPoleEnv
from policies.boltzman import BoltzmannPolicy
from policies.policy_network import PolicyNetwork
from scripts.plot_runs import (plot_files_together, plot_rewards_over_time,
                               plot_rewards_over_time_files, plot_runs)

#########################################
# HYPERPARAMETERS SHARED ACROSS EXPERIMENTS
#
#########################################

actor_critic_hyperparams = {"temperature_decay": [True, False]}

trpo_hyperparams = {"delta": [0.01]}

random_params = {"no_params": [None]}

#########################################
# HYPERPARAMETERS FOR NORMAL EXPERIMENT
#
#########################################
option_critic_hyperparams = {
    "lr": [0.001, 0.0001],
    "gamma": [0.9, 0.99],
    "T": [2],
    "num_options": [2, 4],
    "epsilon": [0.1, 0.2],
    "overall_alpha": [0.0001, 0.001],
}

dqn_hyperparams = {"lr": [0.001, 0.0001], "epsilon": [0.1, 0.2]}

#########################################
# HYPERPARAMETERS FOR STRESS TEST
#
#########################################

option_critic_cartpole_hyperparams_stress = {
    "lr": [0.0001],
    "gamma": [0.9],
    "T": [2],
    "num_options": [2],
    "epsilon": [0.2],
    "overall_alpha": [0.5, 0.0001],
}

option_critic_pacman_hyperparams_stress = {
    "lr": [1e-7],
    "gamma": [0.99, 0.9999],
    "T": [2],
    "num_options": [4],
    "epsilon": [0.1],
    "overall_alpha": [0.0001],
}

dqn_hyperparams_stress = {"lr": [0.001], "epsilon": [0.1, 0.2]}

#########################################
# STRESS TEST CONFIGURATIONS
# For each environment, shared across models.
#########################################
stress_config_cartpole = {
    "gravity": 12,
    "masscart": 0.5,
}

stress_config_pacman = {
    "mode": 4,
}


def create_env(env_name: str):
    if env_name == "CartPole-v1":
        env = MutableCartPoleEnv()
    elif env_name == "Pacman-ram-v5":
        env = MutableAtariEnv(game="pacman", obs_type="ram")
    else:
        raise ValueError(f"Environment {env_name} not recognized.")
    return env


def run_hyperparam_search(
    model: str,
    hyperparams: dict,
    stress_config: list[dict] = None,
    independent_trials: int = 3,
    max_iterations: int = 1000,
    number_of_episodes: int = 1000,
    envs: list = ["CartPole-v1"],
):

    base_seed = 42
    n_seeds = 10

    rng = np.random.default_rng(base_seed)
    seeds = [
        int(s) for s in rng.integers(low=0, high=2**32, size=n_seeds, dtype=np.uint32)
    ]

    if stress_config is not None and len(stress_config) != len(envs):
        raise ValueError(
            "Length of stress_config must match the number of environments."
        )

    for env_num, env_name in enumerate(envs):
        results = {}
        for combo in product(*hyperparams.values()):
            # Zip keys and values into a dictionary for clarity
            config = dict(zip(hyperparams.keys(), combo))
            rewards = np.zeros((independent_trials, number_of_episodes))
            print(config)

            for trial in tqdm(
                range(independent_trials),
                desc=f"Independent runs for {env_name}",
                leave=False,
            ):
                env_obj = create_env(env_name)
                if model == "OptionCritic":
                    agent = OptionCritic(env=env_obj, **config, seed=seeds[trial])
                elif model == "TRPO":
                    policy = PolicyNetwork(env=env_obj)
                    agent = TRPO(
                        env=env_obj, **config, policy=policy, seed=seeds[trial]
                    )
                elif model == "ActorCritic":
                    policy = BoltzmannPolicy(env=env_obj, initial_temperature=1.0)
                    agent = ActorCritic(
                        env=env_obj, **config, policy=policy, seed=seeds[trial]
                    )
                elif model == "RandomAgent":
                    agent = RandomAgent(env=env_obj, seed=seeds[trial])
                elif model == "DQN":
                    agent = DQN(env=env_obj, **config, seed=seeds[trial])
                else:
                    raise ValueError(f"Model {model} not recognized.")

                rewards[trial] = agent.train(
                    stress_config=stress_config[env_num] if stress_config else None,
                    number_of_episodes=number_of_episodes,
                    max_iterations=max_iterations,
                )

            result_name = ""
            for key in config.keys():
                result_name += f"{key}={config[key]},"
            results[result_name] = rewards

            results_serializable = {k: v.tolist() for k, v in results.items()}
            env_name_valid = env_name.replace("/", "_")
            os.makedirs(f"results/{env_name_valid}", exist_ok=True)
            with open(
                f"results/{env_name_valid}/{model}_{env_name_valid}_results.json", "w"
            ) as f:
                json.dump(results_serializable, f)


def main():
    print("Main Experiement ran from here:")
    os.makedirs("results", exist_ok=True)

    models = ["TRPO", "ActorCritic", "RandomAgent", "DQN"]
    params = [
        trpo_hyperparams,
        actor_critic_hyperparams,
        random_params,
        dqn_hyperparams_stress,
    ]
    envs = ["CartPole-v1", "Pacman-ram-v5"]
    stress_configs = [stress_config_cartpole, stress_config_pacman]

    # RUN STRESS TEST ON EACH MODEL
    for model, params in zip(models, params):
        run_hyperparam_search(
            model=model,
            hyperparams=params,
            stress_config=stress_configs,
            independent_trials=10,
            number_of_episodes=1000,
            envs=envs,
        )

    # RUN STRESS TEST ON OPTION CRITIC
    run_hyperparam_search(
        model="OptionCritic",
        hyperparams=option_critic_cartpole_hyperparams_stress,
        stress_config=[stress_config_cartpole],
        independent_trials=10,
        number_of_episodes=1000,
        envs=["CartPole-v1"],
    )

    run_hyperparam_search(
        model="OptionCritic",
        hyperparams=option_critic_pacman_hyperparams_stress,
        stress_config=[stress_config_pacman],
        independent_trials=10,
        number_of_episodes=1000,
        envs=["Pacman-ram-v5"],
    )

    # Plot runs
    plot_files_together(
        [
            "results/CartPole-v1/ActorCritic_CartPole-v1_results.json",
            "results/CartPole-v1/DQN_CartPole-v1_results.json",
            "results/CartPole-v1/OptionCritic_CartPole-v1_results.json",
            "results/CartPole-v1/RandomAgent_CartPole-v1_results.json",
            "results/CartPole-v1/TRPO_CartPole-v1_results.json",
        ]
    )

    # Plot runs
    plot_files_together(
        [
            "results/Pacman-ram-v5/ActorCritic_Pacman-ram-v5_results.json",
            "results/Pacman-ram-v5/DQN_Pacman-ram-v5_results.json",
            "results/Pacman-ram-v5/OptionCritic_Pacman-ram-v5_results.json",
            "results/Pacman-ram-v5/RandomAgent_Pacman-ram-v5_results.json",
            "results/Pacman-ram-v5/TRPO_Pacman-ram-v5_results.json",
        ]
    )


if __name__ == "__main__":
    main()
