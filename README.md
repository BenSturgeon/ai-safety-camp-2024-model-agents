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
```  # backtesting_conversion

```

To get procgen to build on a mac requires some work to get the C code to build.

For this one must install qt@5 using brew and then add this to your system path so you can access it.

Then, normally it makes sense to build procgen from source. I have also had to modify CMakeLists.txt to ensure that warnings are not treated as errors, and this has made building easier.

Change the line in CMAKELISTS.txt that contains the flag -werror to this:
add_compile_options(-Wextra -Wshadow -Wall -Wformat=2 -Wundef -Wvla -Wmissing-include-dirs -Wnon-virtual-dtor -Wno-unused-parameter -Wno-unused-variable)
