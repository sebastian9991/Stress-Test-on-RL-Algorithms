# Stress Test on RL Algorithms

A framework for stress-testing reinforcement learning algorithms under dynamic environment conditions. This project evaluates how various RL algorithms handle sudden changes in environment parameters mid-training, simulating real-world scenarios where operating conditions shift unexpectedly.

## Overview

This repository implements and compares six reinforcement learning algorithms:

- **PPO** (Proximal Policy Optimization)
- **TRPO** (Trust Region Policy Optimization)  
- **Actor-Critic**
- **DQN** (Deep Q-Network)
- **Option-Critic**
- **Random Agent** (baseline)

### Stress Testing Methodology

The stress test methodology involves:
1. Training agents in stable conditions for 500 episodes
2. Introducing environment parameter changes at episode 500
3. Evaluating agent adaptation for 500 additional episodes
4. Comparing performance across algorithms

**Test Environments:**
- **CartPole-v1**: Tests balance control with modified physics parameters

<p align="center">
  <img src="https://github.com/user-attachments/assets/d0022a33-44d7-4769-9fa8-7c475e078192" alt="CartPole Environment" width="600"/>
</p>

- **Pacman (Atari)**: Tests game-playing with altered difficulty modes

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b3543c3-d562-4e34-898f-963304b55d83" alt="Pacman Environment" width="160"/>
</p>

## Getting Started

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip` you can just invoke:

```sh
pip install uv
```

### Installation

```sh
# Clone the repo
git clone git@github.com:sebastian9991/Stress-Test-on-RL-Algorithms.git

# Enter the repo directory
cd Stress-Test-on-RL-Algorithms

# Install core dependencies into an isolated environment
uv sync
```

## Usage

### Running Full Stress-Test Experiments

Run complete experiments across all algorithms and environments:

```sh
./run_experiements.sh
```

Results are saved in `results/` directory with performance plots generated automatically.

## Project Structure

```
├── algorithms/          # RL algorithm implementations
│   ├── ppo.py
│   ├── trpo.py
│   ├── actor_critic.py
│   ├── dqn.py
│   ├── option_critic.py
│   └── random_agent.py
├── mutable_ale/        # Mutable environment wrappers
│   ├── mutable_cartpole.py
│   └── mutable_ALE.py
├── policies/           # Policy implementations
├── scripts/            # Plotting and analysis utilities
├── main.py            # Main experiment runner
└── run_experiements.sh # Convenience script
```

## Results

Experiments generate:
- Reward curves over episodes
- Moving average performance
- Hyperparameter comparison plots
- Per-algorithm performance metrics

Results are saved as JSON files and PNG plots in the `results/` directory.

### Example Results: CartPole Stress Test

The following table shows average rewards per episode for the final 100 episodes across different gravity levels (stress test transitions from 9.8 m/s² to 24.5 m/s² at episode 500):

<p align="center">
  <img src="https://github.com/user-attachments/assets/628326ab-1fc8-4df0-834f-448ee4f04cf1" alt="CartPole Results Table" width="850"/>
</p>

The reward plot below demonstrates the impact of the stress test, showing a clear decrease in rewards immediately after increasing gravity from 9.8 m/s² to 24.5 m/s² at episode 500:

<p align="center">
  <img src="https://github.com/user-attachments/assets/7020d86d-f943-436b-8ad7-8a9b9553e741" alt="CartPole Reward Plot" width="911"/>
</p>

**Note:** The full experimental report is available in `Testing_Generalizatibility_Stress_Test.pdf`.

