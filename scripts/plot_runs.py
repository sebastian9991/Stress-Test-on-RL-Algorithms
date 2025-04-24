import json
from itertools import product
from typing import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_runs(
    files: list[str],
    hyperparams: dict[str, list] = None,
    k=5,
    save=True,
    show=False,
    average_over=50,
) -> None:
    """Plot the runs from several json files
    Args:
        files (list): A list of json files to plot.
        hyperparams (dict): A dictionary of hyperparameters and their values. MUST be the same as what was used to generate the json files.
            Include hyperparams to plot the performance as a function of the hyperparameters.
        k (int): The number of top performers to plot.
        save (bool): Whether to save the plot.
        show (bool): Whether to show the plot.
    """
    ### For each hyperparameter combination, plot the performance over time ###
    plot_rewards_over_time_files(files, show=show, save=save, average_over=average_over)

    # ### For each hyperparameter, plot the performance in the last 100 episodes as a function of the hyperparameter values ###
    if hyperparams is not None:
         plot_hyperparam_performance(files, hyperparams, show=show, save=save)

    ### For the top k hyperparameter combinations (as defined as the highest average reward over the last 100 episodes), plot the performance over time ###
    plot_top_k_rewards_over_time(
        files, k=k, show=show, save=save, average_over=average_over
    )

def plot_files_together(
        files: list[str],
        show: bool = True,
        save: bool = False,
        average_over=50,
        title: str = "Combined",
    ) -> None:
        """Plot the rewards over time for each hyperparameter combination in a set of files
        Args:
        files (list): A list of json files to plot.
        show (bool): Whether to show the plot.
        save (bool): Whether to save the plot.
        average_over (int): The number of episodes to average over for the moving average plot.
        """
        rewards = {}
        for file in files:
            with open(file) as f:
                file_rewards = json.load(f)
                # Add a preamble to the labels
                file_rewards = {f"{file[8:-5]}_{key}": value for key, value in file_rewards.items()}
                rewards.update(file_rewards)
        plot_rewards_over_time(
        rewards, show=show, save=save, average_over=average_over, file=title + ".json"
        )


def plot_hyperparam_performance(
    files: list[str],
    hyperparams: dict[str, list],
    show: bool = True,
    save: bool = False,
) -> None:
    """Plot the performance of models as a function of each hyperparameter.
    Args:
        files (list): A list of json files to plot.
        hyperparams (dict): A dictionary of hyperparameters and their values. MUST be the same as what was used to generate the json files.
        show (bool): Whether to show the plot.
        save (bool): Whether to save the plot.
    """
    ordered_hyperparams = OrderedDict(hyperparams)
    counter = 0
    for file in files:
        rewards = load_rewards(file)
        num_plots = len(ordered_hyperparams)
        fig, axs = plt.subplots(num_plots // 2, 2, figsize=(7, 10))
        fig.suptitle(f"Performance as a function of hyperparameters for \n{file[:-5]}")

        # For each hyperparameter, make a new plot
        for plot_num, hyperparam in enumerate(hyperparams):

            # Ticks of the x axis
            values = sorted(hyperparams[hyperparam])

            # Where to insert back in the value for reading dict
            location = list(ordered_hyperparams.keys()).index(hyperparam)
            # print(location)
            partial_dict = hyperparams.copy()
            partial_dict.pop(hyperparam)
            # print(partial_dict)

            values_product = list(product(*partial_dict.values()))

            # REMOVE COLOURS IF WANT TO SOLVE BUG
            num_keys = len(values_product)
            color_indices = np.linspace(0, 1, num_keys)
            colors = plt.cm.rainbow(color_indices)

            for color_id, combo in enumerate(values_product):
                partial_hyperparam_set = dict(zip(partial_dict.keys(), combo))

                all_else_fixed = []
                placed = False  # If the hyperparam is the LAST one, this is needed.
                for value in values:
                    # Make a new dictionary with the hyperparameter set to the value
                    hyperparam_dict = partial_hyperparam_set.copy()
                    result_name = ""
                    for spot, key in enumerate(hyperparam_dict.keys()):
                        if spot == location:
                            result_name += f"{hyperparam}={value},"
                            result_name += f"{key}={hyperparam_dict[key]},"
                            placed = True
                        else:
                            result_name += f"{key}={hyperparam_dict[key]},"
                    if not placed:
                        result_name += f"{hyperparam}={value},"
                    all_else_fixed.append(result_name)
                counter += 1

                # For each of those sets, plot the performance of the models as a function of that hyperparameter

                # print(len(rewards[all_else_fixed[0]]), len(rewards[all_else_fixed[0]][0]))
                performances = np.zeros(len(all_else_fixed))
                for k, data in enumerate(all_else_fixed):
                    # Get the average reward over the last 100 episodes
                    try:
                        means = np.mean(rewards[data], axis=0)
                        performances[k] = np.mean(means[-100:])
                    except:
                        print(
                            f"Data missing for {hyperparam}. Data not found for {data}"
                        )
                        performances[k] = None

                axs[plot_num // 2, plot_num % 2].plot(
                    values,
                    performances,
                    color=colors[color_id],
                )
                axs[plot_num // 2, plot_num % 2].set_title(
                    f"Performance as a function of {hyperparam}", fontsize=8
                )
                axs[plot_num // 2, plot_num % 2].set_xlabel(hyperparam)
                axs[plot_num // 2, plot_num % 2].set_ylabel("Reward")
                axs[plot_num // 2, plot_num % 2].set_xticks(values)
                if len(set(values)) > 1:
                    lr = stats.linregress(values, range(0, len(values)))
                    if lr.rvalue < 0.5:
                        axs[plot_num // 2, plot_num % 2].set_xscale("log")
        fig.tight_layout(pad=1.2)
        if save:
            plt.savefig(f"{file[:-5]}_hyperparam_performance.png")
        if show:
            plt.show()


def load_rewards(file: list[str]) -> dict[str, list]:
    """Load the rewards from a json file
    Args:
        file (str): The name of the json file to load.
    Returns:
        rewards (dict): A dictionary of rewards for each hyperparameter combination."""
    try:
        with open(file) as f:
            rewards = json.load(f)
    except:
        print(f"{file} not found")
    return rewards


def plot_rewards_over_time_files(
    files: list[str],
    show: bool = True,
    save: bool = False,
    average_over=50,
) -> None:
    """Plot the rewards over time for each hyperparameter combination in a set of files
    Args:
        files (list): A list of json files to plot.
        show (bool): Whether to show the plot.
        save (bool): Whether to save the plot.
        average_over (int): The number of episodes to average over for the moving average plot.
    """
    for file in files:
        rewards = load_rewards(file)
        plot_rewards_over_time(
            rewards, show=show, save=save, average_over=average_over, file=file
        )


def plot_top_k_rewards_over_time(
    files: list[str],
    k: int = 5,
    show: bool = True,
    save: bool = False,
    average_over=50,
) -> None:
    """Plot the rewards over time for the top k hyperparameter combinations in a set of files. We define the top
    k hyperparameter combinations as the ones with the highest average reward over the last 100 episodes. "
    Args:
       files (list): A list of json files to plot.
       k (int): The number of top performers to plot.
       show (bool): Whether to show the plot.
       save (bool): Whether to save the plot.
       average_over (int): The number of episodes to average over for the moving average plot.
    """
    for file in files:
        rewards = load_rewards(file)
        top_k_performers = find_top_k_performers(rewards, k=k)
        top_k_rewards = {}
        for i in top_k_performers:
            hyperparams = list(rewards.keys())[i]
            top_k_rewards[hyperparams] = rewards[hyperparams]

        plot_rewards_over_time(
            top_k_rewards,
            show=show,
            save=save,
            average_over=average_over,
            file=file[:-5] + "_top_k_rewards.json",
        )


def find_top_k_performers(rewards: dict[str, list], k: int = 5) -> list:
    """Find the top k performers based on the average reward over the last 100 episodes.
     Args:
        rewards (dict): A dictionary of rewards for each hyperparameter combination.
        k (int): The number of top performers to return.

    Returns:
        list: A list of the indices of top k performers, sorted by average reward."""
    avg_rewards = np.zeros(len(rewards.keys()))
    for i, hyperparams in enumerate(rewards):
        means = np.mean(rewards[hyperparams], axis=0)
        avg_rewards[i] = np.mean(means[-100:])
    # print(avg_rewards)

    # Sort the hyperparameters by average reward
    sorted_hyperparams = np.argsort(-avg_rewards)

    # Get the top k performers
    top_k_performers = sorted_hyperparams[:k]

    return top_k_performers


def plot_rewards_over_time(
    rewards: dict[str, list],
    show: bool = True,
    save: bool = False,
    average_over=50,
    file: str = "rewards.json",
) -> None:
    """Plot the rewards over time for each hyperparameter combination.
    Args:
        rewards (dict): A dictionary of rewards for each hyperparameter combination.
        show (bool): Whether to show the plot.
        save (bool): Whether to save the plot.
        average_over (int): The number of episodes to average over for the moving average plot.
        file (str): The name of the file to save the plot to.
    """

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Performance over time for \n{file[:-5]}")

    # Divide by 2 since the window goes before and after the current episode
    average_over = average_over // 2

    # Set up the colours
    keys = list(rewards.keys())
    num_keys = len(keys)
    color_indices = np.linspace(0, 1, num_keys)
    colors = plt.cm.rainbow(color_indices)

    for i, hyperparams in enumerate(rewards):
        # Plot the rewards
        # Set label to the hyperparameters
        means = np.mean(rewards[hyperparams], axis=0)
        stds = np.std(rewards[hyperparams], axis=0)

        axs[0, 0].plot(means, label=hyperparams, color=colors[i])
        axs[0, 0].axvline(x=500, color='red', linestyle='--', linewidth=1) #For the stress-test location
        axs[0, 0].set_title("Reward")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Reward")
        axs[0, 0].fill_between(
            range(len(means)), means - stds, means + stds, alpha=0.1, color=colors[i]
        )
        # axs[0,0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        # Plot the cumulative reward
        cumulative = np.cumsum(means)
        std_devs = np.cumsum(stds)
        axs[1, 0].plot(cumulative, label=hyperparams, color=colors[i])
        axs[1, 0].set_title("Cumulative Reward")
        axs[1, 0].set_xlabel("Episode")
        axs[1, 0].set_ylabel("Reward")
        axs[1, 0].fill_between(
            range(len(cumulative)),
            cumulative - std_devs,
            cumulative + std_devs,
            alpha=0.1,
            color=colors[i],
        )
        # axs[1,0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        # Plot the moving average
        moving_average = []
        avg_std_dev = []
        for j in range(len(means)):
            moving_average.append(
                np.mean(
                    means[max(0, j - average_over) : min(len(means), j + average_over)]
                )
            )
            avg_std_dev.append(
                np.mean(
                    stds[max(0, j - average_over) : min(len(means), j + average_over)]
                )
            )

        axs[0, 1].plot(moving_average, label=hyperparams, color=colors[i])
        axs[0, 1].set_title("Moving Average")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Reward")
        axs[0, 1].fill_between(
            range(len(moving_average)),
            np.array(moving_average) - np.array(avg_std_dev),
            np.array(moving_average) + np.array(avg_std_dev),
            alpha=0.1,
            color=colors[i],
        )
        # axs[0,1].legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        # Plot the average reward over the last 100 episodes
        avg100 = np.mean(means[-100:])
        axs[1, 1].bar(i, avg100, color=colors[i])
        axs[1, 1].set_title("Average Reward over Last 100 Episodes")
        axs[1, 1].set_ylabel("Reward")
        axs[1, 1].set_xlabel("Hyperparameters")

    axs[1, 1].set_xticks(range(0, num_keys, 2))
    handles, labels = axs[0, 0].get_legend_handles_labels()

    numbered_labels = []
    for i, label in enumerate(labels):
        numbered_label = f"{i}" + ": " + label
        if len(label) > 80:
            numbered_label = label[:80] + "\n" + label[80:]
        numbered_labels.append(numbered_label)
    fig.tight_layout(pad=1.2, rect=[0, 0, 0.7, 1])
    fig.legend(
        handles,
        numbered_labels,
        loc="lower left",
        bbox_to_anchor=(0.7, 0.3),
        fontsize=6,
    )

    fig.set_size_inches(14, 8)
    if save:
        names = [
            "reward",
            "cumulative_reward",
            "moving_average",
            "avg_reward_last100"
        ]
        for ax, name in zip(axs.flatten(), names):
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f"{file[:-5]}_{name}.png", bbox_inches=extent)
        plt.savefig(f"{file[:-5]}_full.png")

    if show:
        plt.show()
    plt.close(fig)

# plot_files_together(["../results/CartPole-v1/PPO_CartPole-v1_results_02.json"])