# Save results dict as a json file
import json
import os
from itertools import product
from unittest import result

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from algorithms.actor_critic import ActorCritic
from algorithms.option_critic import OptionCritic
from algorithms.random_agent import RandomAgent
from algorithms.trpo import TRPO
from policies.boltzman import BoltzmannPolicy
from policies.policy_network import PolicyNetwork
from scripts.plot_runs import (plot_rewards_over_time,
                               plot_rewards_over_time_files, plot_runs)

VALID_MODELS = ["OptionCritic", "ActorCritic", "TRPO", "RandomAgent"]

option_critic_hyperparams = {
    "lr": [0.001, 0.0001],
    "gamma": [0.1, 0.01],
    "T": [2],
    "num_options": [2, 4],
    "epsilon": [0.1, 0.2],
    "overall_alpha": [0.0001, 0.001],
}
actor_critic_hyperparams = {"temperature_decay": [True, False]}

trpo_hyperparams = {"delta": [0.01]}

random_params = {"no_params": [None]}


def run_hyperparam_search(
    model: str,
    hyperparams: dict,
    independent_trials: int = 3,
    max_iterations: int = 1000,
    number_of_episodes: int = 1000,
    envs: list = ["CartPole-v1"],
):

    if model not in VALID_MODELS:
        raise ValueError(
            f"Model must be in {VALID_MODELS}, but got {model}. Add {model} to the function."
        )

    for env_name in envs:
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
                env = gym.make(env_name)

                if model == "OptionCritic":
                    agent = OptionCritic(env=env, **config, seed=trial)
                elif model == "TRPO":
                    policy = PolicyNetwork(env=env)
                    agent = TRPO(env=env, **config, policy=policy, seed=trial)
                elif model == "ActorCritic":
                    policy = BoltzmannPolicy(env=env, initial_temperature=1.0)
                    agent = ActorCritic(env=env, **config, policy=policy, seed=trial)
                elif model == "RandomAgent":
                    agent = RandomAgent(env=env)

                rewards[trial] = agent.train(number_of_episodes, max_iterations)

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
        independent_trials=3,
        number_of_episodes=10,
        envs=["CartPole-v1"],
    )
    plot_runs(
        files=["results/TRPO_CartPole-v1_results.json"],
        hyperparams=trpo_hyperparams,
        show=True,
    )

    run_hyperparam_search(
        model="ActorCritic",
        hyperparams=actor_critic_hyperparams,
        independent_trials=3,
        number_of_episodes=10,
        envs=["CartPole-v1"],
    )
    plot_runs(
        files=["results/ActorCritic_CartPole-v1_results.json"],
        hyperparams=actor_critic_hyperparams,
        show=True,
    )

    run_hyperparam_search(
        model="RandomAgent",
        hyperparams=random_params,
        independent_trials=3,
        number_of_episodes=10,
        envs=["CartPole-v1"],
    )
    plot_runs(
        files=["results/RandomAgent_CartPole-v1_results.json"],
        hyperparams=random_params,
        show=True,
    )


if __name__ == "__main__":
    main()
