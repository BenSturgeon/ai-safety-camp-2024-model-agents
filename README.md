# AI Safety Camp - Model Agents

Research on how to both identify optimization targets in model agents, and how to robustly influence agent behavior based on modifying those targets.

## Prerequisites
- Python 3.9 (specific version required)
- Qt5 (for macOS users)
- Git

## Installation Options

### Option 1: Using uv (Recommended for Speed)
```bash
# Install uv if you haven't already
pip install uv

# Create virtual environment with Python 3.9
uv venv --python=3.9

# Activate the environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install procgen from source (required)
git clone https://github.com/openai/procgen.git
cd procgen
git checkout 1d6dcd9f8ac5544c0f99d3b82c3a5ecb27e481f6  # version 0.10.7
python -m pip install -e .

# Install procgen-tools
python -m pip install git+https://github.com/UlisseMini/procgen-tools.git
```

### Option 2: Using poetry
```bash
# Install poetry if you haven't already
pip install poetry

# Configure poetry to use Python 3.9
poetry env use python3.9

# Install dependencies
poetry install

# Enter poetry shell
poetry shell

# Install procgen from source (required)
git clone https://github.com/openai/procgen.git
cd procgen
git checkout 1d6dcd9f8ac5544c0f99d3b82c3a5ecb27e481f6  # version 0.10.7
python -m pip install -e .

# Install procgen-tools
python -m pip install git+https://github.com/UlisseMini/procgen-tools.git
```

### Option 3: Using pip
```bash
# Create virtual environment with Python 3.9
python3.9 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
python -m pip install -r requirements.txt

# Install procgen from source (required)
git clone https://github.com/openai/procgen.git
cd procgen
git checkout 1d6dcd9f8ac5544c0f99d3b82c3a5ecb27e481f6  # version 0.10.7
python -m pip install -e .

# Install procgen-tools
python -m pip install git+https://github.com/UlisseMini/procgen-tools.git
```

## macOS Setup
macOS users need to install Qt5 before building procgen:

```bash
# Install Qt5
brew install qt@5

# Add Qt to your PATH (add this to your ~/.zshrc or ~/.bashrc)
export PATH="/usr/local/opt/qt@5/bin:$PATH"
```

## Known Issues and Solutions

### Procgen Build Issues
If you encounter build errors with procgen, modify `CMakeLists.txt` in the procgen source:
```cmake
# Find this line:
# add_compile_options(-Wextra -Wshadow -Wall -Wformat=2 -Wundef -Wvla -Wmissing-include-dirs -Wnon-virtual-dtor -Werror)

# Replace it with:
add_compile_options(-Wextra -Wshadow -Wall -Wformat=2 -Wundef -Wvla -Wmissing-include-dirs -Wnon-virtual-dtor -Wno-unused-parameter -Wno-unused-variable)
```

### Environment Issues
- Always use `python -m pip` instead of just `pip` when not using uv or poetry
- Verify your environment is correct:
```bash
which python
python --version  # should show Python 3.9.x
```

### Version Conflicts
- Use a fresh virtual environment if you encounter persistent issues
- Make sure procgen is installed before procgen-tools
- Verify you're using Python 3.9

## Verifying Installation
After installation, verify everything works:
```python
python
>>> import procgen_tools
>>> import procgen
```

## Project Structure
```
.
├── requirements.txt          # Core dependencies
├── pyproject.toml           # Poetry configuration
├── .python-version          # Python version file
└── .gitignore              # Includes virtual environments and build artifacts
```