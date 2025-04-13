# AI Safety Camp - Model Agents

Research on how to both identify optimization targets in model agents, and how to robustly influence agent behavior based on modifying those targets.

## Prerequisites
- Python 3.9 (specific version required)
- Qt5 (for macOS users)
- Git

## Installation Options
### Option 1: Using a script (Recommended for Speed)

We provide an automated setup script to install all dependencies, configure your environment, and prepare everything for running experiments:

This is designed for running on a remote linux machine and includes steps for generating an ssh key and installing procgen and procgen-tools from source.

It uses UV and will install it automatically. It will also install QT5 which is a necessary dependency for building Procgen. This is needed on both mac and Linux.

```bash
chmod +x automated_setup.sh

./automated_setup.sh
```

### Option 2: Manual installation using uv 

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
uv  pip install -e .


# Install procgen tools from source (required)
git clone https://github.com/UlisseMini/procgen-tools
cd procgen-tools
uv -m pip install -e .


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
