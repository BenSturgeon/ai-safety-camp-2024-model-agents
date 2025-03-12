#!/bin/bash
set -euo pipefail

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Setting up development environment..."

# Install required packages
if command_exists apt-get; then
    sudo apt-get update
    sudo apt-get install -y git cmake \
        qtbase5-dev qttools5-dev qttools5-dev-tools \
        qtbase5-dev-tools libqt5opengl5-dev \
        build-essential g++
elif command_exists yum; then
    sudo yum update -y
    sudo yum install -y git cmake qt5-devel gcc-c++ make
fi

# Check and create SSH key if needed
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "Creating SSH key..."
    ssh-keygen -t ed25519 -C "vast-instance-temp" -f ~/.ssh/id_ed25519 -N ""
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/id_ed25519
else
    echo "SSH key already exists, skipping creation..."
fi

# Display public key
echo -e "\nAdd this public key to your Git provider (GitHub/GitLab):"
echo "================ PUBLIC KEY START ================"
cat ~/.ssh/id_ed25519.pub
echo "================= PUBLIC KEY END ================="

# Pause to allow copying the SSH key
echo -e "\nPausing to allow time to copy the SSH key..."
echo "Press Enter once you have copied the key to continue..."
read -r


# Configure Git if not already configured
echo -e "\nChecking Git configuration..."
if [ -z "$(git config --global user.email)" ] || [ -z "$(git config --global user.name)" ]; then
    echo "Git configuration not found. Please configure:"
    read -p "Enter your Git email: " git_email
    read -p "Enter your Git username: " git_username
    git config --global user.email "$git_email"
    git config --global user.name "$git_username"
else
    echo "Git already configured:"
    echo "Email: $(git config --global user.email)"
    echo "Username: $(git config --global user.name)"
fi

# Clone the main repository (ai-safety-camp-2024-model-agents)
REPO_DIR="ai-safety-camp-2024-model-agents"
if [ ! -d "$REPO_DIR" ]; then
    git clone git@github.com:BenSturgeon/ai-safety-camp-2024-model-agents.git
else
    echo "Repository directory already exists, pulling latest changes..."
    (cd "$REPO_DIR" && git pull)
fi

# Set up Python virtual environment using uv
echo "Setting up Python environment..."
pip install uv

# Install uv using the provided install script
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is in PATH (adjust the path if necessary)
export PATH="$HOME/.cargo/bin:$PATH"

# Upgrade pip and create a virtual environment with Python 3.9
uv pip install --upgrade pip

cd "$REPO_DIR"

uv venv -p 3.9 .venv
curl -L -o model_interpretable.pt "https://drive.google.com/uc?export=download&id=1oYsWk3hhObafDA29x0YDcK9h6LQmYDNr"
# Activate the virtual environment and install Python dependencies from the cloned repo
source .venv/bin/activate
uv pip install -r "$REPO_DIR/requirements.txt"

# Set Qt5 paths for procgen build
echo "Setting up Qt5 environment variables..."
if [ -d "/opt/conda/envs/procgen/lib/cmake/Qt5" ]; then
    export Qt5_DIR="/opt/conda/envs/procgen/lib/cmake/Qt5"
    export CMAKE_PREFIX_PATH="/opt/conda/envs/procgen/lib/cmake/Qt5"
elif [ -d "/usr/lib/x86_64-linux-gnu/cmake/Qt5" ]; then
    export Qt5_DIR="/usr/lib/x86_64-linux-gnu/cmake/Qt5"
    export CMAKE_PREFIX_PATH="/usr/lib/x86_64-linux-gnu/cmake/Qt5"
fi

# Function to search for Qt5Config.cmake if needed
find_qt5_config() {
    echo "Searching for Qt5Config.cmake..."
    qt5_config=$(find / -name "Qt5Config.cmake" 2>/dev/null | head -n 1)
    if [ -n "$qt5_config" ]; then
        qt5_dir=$(dirname "$qt5_config")
        export Qt5_DIR="$qt5_dir"
        export CMAKE_PREFIX_PATH="$qt5_dir"
        echo "Found Qt5 configuration at: $qt5_dir"
    else
        echo "Warning: Could not find Qt5Config.cmake"
    fi
}

if [ -z "${Qt5_DIR:-}" ]; then
    find_qt5_config
fi

cd ..

# Clone and install procgen from source
echo "Installing procgen from source..."
if [ ! -d "procgen" ]; then
    git clone https://github.com/openai/procgen.git
else
    echo "Procgen directory already exists, pulling latest changes..."
    (cd procgen && git pull && uv pip install -e .)
fi

# Clone and install procgen-tools
echo "Installing procgen-tools..."
if [ ! -d "procgen-tools" ]; then
    git clone https://github.com/UlisseMini/procgen-tools.git
    (cd procgen-tools && uv pip install -e .)
else
    echo "Procgen-tools already installed, upgrading to latest version..."
    (cd procgen-tools && git pull && uv pip install --upgrade -e .)
fi

echo -e "\nSetup complete! Remember to:"
echo "1. Add the public key shown above to your Git provider"
echo "2. Remove the key when you're done with this instance"
echo "3. The virtual environment is activated and ready to use"

echo -e "\nImportant notes for procgen:"
echo "1. If you encounter 'Illegal instruction' errors, try setting:"
echo "   export MUJOCO_GL=osmesa"
echo "   export MESA_GL_VERSION_OVERRIDE=3.3"
echo "2. If you get Qt5 errors during installation, verify Qt5_DIR is set correctly:"
echo "   echo \$Qt5_DIR"
echo "3. For render issues, try different backend options:"
echo "   render_mode=\"rgb_array\" or without render_mode"

echo -e "\nTesting imports..."
python -c "
import os
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MUJOCO_GL'] = 'osmesa'
import procgen
import procgen_tools
print('Import test successful!')
"
