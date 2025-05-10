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

_Full End-to-End Experiments_

```sh
./run_experiements.sh
```

