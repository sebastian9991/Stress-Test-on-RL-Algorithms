from gettext import find
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_runs(files: list[str], hyperparams: dict[str, list], k = 5, save = True, show = False):
    """ Plot the runs from several json files"""

    plot_rewards_over_time_files(files, show=show, save=save)



    ### For each hyperparameter, plot the performance in the last 100 episodes as a function of the hyperparameter values ###


    ### For the top 5 hyperparameter combinations (as defined as the highest average reward over the last 100 episodes), plot the performance over time ###
    plot_top_k_rewards_over_time(files, k=k, show=show, save=save)


    
def load_rewards(file: list[str]) -> dict[str, list]:
    """ Load the rewards from a json file """
    try:
        with open(file) as f:
            rewards = json.load(f)
    except:
        print(f"{file} not found")
    return rewards

def plot_rewards_over_time_files(files: list[str], show: bool = True, save: bool = False) -> None:
    """ Plot the rewards over time for each hyperparameter combination in a set of files """
    for file in files:
        rewards = load_rewards(file)
        plot_rewards_over_time(rewards, show=show, save=save, file=file)

def plot_top_k_rewards_over_time(files: list[str], k: int = 5, show: bool = True, save: bool = False) -> None:
    """ Plot the rewards over time for the top k hyperparameter combinations in a set of files. We define the top
     k hyperparameter combinations as the ones with the highest average reward over the last 100 episodes. """
    for file in files:
        rewards = load_rewards(file)
        top_k_performers = find_top_k_performers(rewards, k=k)
        top_k_rewards = {}
        for i in top_k_performers:
            hyperparams = list(rewards.keys())[i]
            top_k_rewards[hyperparams] = rewards[hyperparams]
        
        plot_rewards_over_time(top_k_rewards, show=show, save=save, file=file)

def find_top_k_performers(rewards: dict[str, list], k: int = 5) -> list:
    """ Find the top k performers based on the average reward over the last 100 episodes.
     Args:
        rewards (dict): A dictionary of rewards for each hyperparameter combination.
        k (int): The number of top performers to return.
        
    Returns:
        list: A list of the indices of top k performers, sorted by average reward."""
    avg_rewards = np.zeros(len(rewards.keys()))
    for i, hyperparams in enumerate(rewards):
        means = np.mean(rewards[hyperparams], axis=0)
        avg_rewards[i] = np.mean(means[-100:])
    print(avg_rewards)

    # Sort the hyperparameters by average reward
    sorted_hyperparams = np.argsort(-avg_rewards)

    # Get the top k performers
    top_k_performers = sorted_hyperparams[:k]

    return top_k_performers

def plot_rewards_over_time(rewards: dict[str, list], show: bool = True, save: bool = False, file: str = "rewards.json") -> None:
    """ Plot the rewards over time for each hyperparameter combination.
    Args:
        rewards (dict): A dictionary of rewards for each hyperparameter combination.
        show (bool): Whether to show the plot.
        save (bool): Whether to save the plot.
        file (str): The name of the file to save the plot to.
    """
    
    fig, axs = plt.subplots(2,2, figsize=(10,7))

    # Set up the colours
    keys = list(rewards.keys())
    num_keys = len(keys)
    color_indices = np.linspace(0, 1, num_keys)
    colors = plt.cm.rainbow(color_indices)

    for i, hyperparams in enumerate(rewards):
        #Plot the rewards
        # Set label to the hyperparameters
        means = np.mean(rewards[hyperparams], axis=0)
        stds = np.std(rewards[hyperparams], axis=0)

        axs[0,0].plot(means, label = hyperparams, color = colors[i])
        axs[0,0].set_title("Reward")
        axs[0,0].set_xlabel("Episode")
        axs[0,0].set_ylabel("Reward")
        axs[0,0].fill_between(range(len(means)), means-stds, means+stds, alpha=0.1, color = colors[i])
        #axs[0,0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))


        #Plot the cumulative reward
        cumulative = np.cumsum(means)
        std_devs = np.cumsum(stds)
        axs[1,0].plot(cumulative, label = hyperparams, color = colors[i])
        axs[1,0].set_title("Cumulative Reward")
        axs[1,0].set_xlabel("Episode")
        axs[1,0].set_ylabel("Reward")
        axs[1,0].fill_between(range(len(cumulative)), cumulative-std_devs, cumulative+std_devs, alpha=0.1, color = colors[i])
        #axs[1,0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        #Plot the moving average
        moving_average = []
        avg_std_dev = []
        average_over = 20
        for j in range(len(means)):
            moving_average.append(np.mean(means[max(0,j-average_over):min(len(means),j+average_over)]))
            avg_std_dev.append(np.mean(stds[max(0,j-average_over):min(len(means),j+average_over)]))

        axs[0,1].plot(moving_average, label = hyperparams, color = colors[i])
        axs[0,1].set_title("Moving Average")
        axs[0,1].set_xlabel("Episode")
        axs[0,1].set_ylabel("Reward")
        axs[0,1].fill_between(range(len(moving_average)), np.array(moving_average)-np.array(avg_std_dev), np.array(moving_average)+np.array(avg_std_dev), alpha=0.1, color = colors[i])
        #axs[0,1].legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        #Plot the average reward over the last 100 episodes
        avg100 = np.mean(means[-100:])
        axs[1,1].bar(i, avg100, color = colors[i])
        axs[1,1].set_title("Average Reward over Last 100 Episodes")
        axs[1,1].set_ylabel("Reward")
        axs[1,1].set_xlabel("Hyperparameters") 
    
    axs[1,1].set_xticks(range(0,num_keys,2))
    handles, labels = axs[0,0].get_legend_handles_labels()

    numbered_labels = []
    for i, label in enumerate(labels):
        numbered_labels.append(f"{i}" + ': ' + label)
    fig.tight_layout(pad=1.2, rect=[0, 0, 0.7, 1])
    fig.legend(handles, numbered_labels, loc='lower left', bbox_to_anchor=(0.7, 0.3), fontsize=6 )


    fig.set_size_inches(14, 8)

    if save:
        plt.savefig(f"{file[:-5]}.png")
    if show:
        plt.show()
    plt.close(fig)

