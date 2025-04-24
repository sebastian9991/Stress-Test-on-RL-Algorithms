
import numpy as np
import json

def last100_perf(
        files: list[str],
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
        last100_perf_per_file(
        rewards
        )

def last100_perf_per_file(
    rewards: dict[str, list],
) -> None:
    """Plot the rewards over time for each hyperparameter combination.
    Args:
        rewards (dict): A dictionary of rewards for each hyperparameter combination.
        show (bool): Whether to show the plot.
        save (bool): Whether to save the plot.
        average_over (int): The number of episodes to average over for the moving average plot.
        file (str): The name of the file to save the plot to.
    """

    for i, hyperparams in enumerate(rewards):
        # Plot the rewards
        # Set label to the hyperparameters
        means = np.mean(rewards[hyperparams], axis=0)

        avg100 = np.mean(means[-100:])
        std100 = np.std(means[-100:])

        print(
            f"Hyperparameters: {hyperparams}, Avg last 100: {avg100:.2f} +/- {std100:.2f}"
        )
 