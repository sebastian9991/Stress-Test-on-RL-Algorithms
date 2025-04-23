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
git clone 

# Enter the repo directory
cd Stress-Test-on-RL-Algorithms

# Install core dependencies into an isolated environment
uv sync

# [Optional] Install extra dependencies to run analytics
uv sync --group analytics
```

## Usage

### Running Plausibility Vaccine

_Full End-to-End Experiments_

```sh
./run_experiements.sh
```
