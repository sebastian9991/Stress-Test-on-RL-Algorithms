# Save results dict as a json file
import json
import os
from itertools import product
from unittest import result

import gymnasium as gym
import matplotlib.pyplot as plt
from mutable_ale.mutable_ALE import MutableAtariEnv
import numpy as np
from tqdm import tqdm

from algorithms.actor_critic import ActorCritic
from algorithms.option_critic import OptionCritic
from algorithms.random_agent import RandomAgent
from algorithms.trpo import TRPO
from mutable_ale.mutable_cartpole import MutableCartPoleEnv
from policies.boltzman import BoltzmannPolicy
from policies.policy_network import PolicyNetwork
from scripts.plot_runs import (plot_rewards_over_time,
                               plot_rewards_over_time_files, plot_runs)

VALID_MODELS = ["OptionCritic", "ActorCritic", "TRPO", "RandomAgent"]

option_critic_hyperparams = {
    "lr": [ 0.001, 0.0001],
    "gamma": [0.9, 0.99],
    "T": [2],
    "num_options": [2, 4],
    "epsilon": [0.1, 0.2],
    "overall_alpha": [0.0001, 0.001],
}
actor_critic_hyperparams = {"temperature_decay": [True, False]}

trpo_hyperparams = {"delta": [0.01]}

random_params = {"no_params": [None]}

##Stress-Test config

stress_config_cartpole = {
    "gravity": 12,
    "masscart": 0.65,
}

stress_config_pacman = {
    "mode": 4,
}


def run_hyperparam_search(
    model: str,
    hyperparams: dict,
    stress_config: dict,
    independent_trials: int = 3,
    max_iterations: int = 1000,
    number_of_episodes: int = 1000,
    envs: list = ["CartPole-v1"],
):

    if model not in VALID_MODELS:
        raise ValueError(
            f"Model must be in {VALID_MODELS}, but got {model}. Add {model} to the function."
        )

    for env_dict in envs:
        env_name, env_obj = next(iter(env_dict.items()))
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

                if model == "OptionCritic":
                    agent = OptionCritic(env=env_obj, **config, seed=trial)
                elif model == "TRPO":
                    policy = PolicyNetwork(env=env_obj)
                    agent = TRPO(env=env_obj, **config, policy=policy, seed=trial)
                elif model == "ActorCritic":
                    policy = BoltzmannPolicy(env=env_obj, initial_temperature=1.0)
                    agent = ActorCritic(env=env_obj, **config, policy=policy, seed=trial)
                elif model == "RandomAgent":
                    agent = RandomAgent(env=env_obj)

                rewards[trial] = agent.train(
                    stress_config=stress_config,
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
    env_cart = {"CartPole-v1": MutableCartPoleEnv()}
    env_pacman = {"Pacman-ram-v5": MutableAtariEnv(game = "pacman", obs_type = 'ram')}

    # run_hyperparam_search(
    #     model="OptionCritic",
    #     hyperparams=option_critic_hyperparams,
    #     independent_trials=3,
    #     number_of_episodes=10,
    #     envs=["CartPole-v1"],
    # )
    # plot_runs(
    #     files=["results/OptionCritic_CartPole-v1_results.json"],
    #     hyperparams=option_critic_hyperparams,
    #     show=True,
    # )

    run_hyperparam_search(
        model="TRPO",
        hyperparams=trpo_hyperparams,
        stress_config=stress_config_cartpole,
        independent_trials=3,
        number_of_episodes=1000,
        envs=[env_cart],
    )
    plot_runs(
        files=["results/CartPole-v1/TRPO_CartPole-v1_results.json"],
        hyperparams=trpo_hyperparams,
        show=True,
    )

    # run_hyperparam_search(
    #     model="ActorCritic",
    #     hyperparams=actor_critic_hyperparams,
    #     independent_trials=3,
    #     number_of_episodes=10,
    #     envs=[env_cart],
    # )
    # plot_runs(
    #     files=["results/ActorCritic_CartPole-v1_results.json"],
    #     hyperparams=actor_critic_hyperparams,
    #     show=True,
    # )
    #
    # run_hyperparam_search(
    #     model="RandomAgent",
    #     hyperparams=random_params,
    #     independent_trials=3,
    #     number_of_episodes=10,
    #     envs=[env_cart],
    # )
    # plot_runs(
    #     files=["results/RandomAgent_CartPole-v1_results.json"],
    #     hyperparams=random_params,
    #     show=True,
    # )
    #


if __name__ == "__main__":
    main()
