# AI Safety Camp - Model Agents
Research on how to both identify optimization targets in model agents, and how to robustly influence agent behavior based on modifying those targets.

## Setup
Install dependencies using poetry:
```bash
poetry install
```

If poetry is not installed, you can create a virtual environment and install the dependencies manually:
```bash
python3.9 -m venv venv
source venv/bin/activate
poetry install
```

Since `procgen-tools` is "special" (deps are not managed well) we need to install it manually afterwards:
```bash
(to access the poetry environment in your shell for installing pip, use the command: "poetry shell")
pip install git+https://github.com/UlisseMini/procgen-tools.git
``` # backtesting_conversion
