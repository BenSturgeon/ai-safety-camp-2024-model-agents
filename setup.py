from setuptools import setup, find_packages

setup(
    name="ai-safety-camp",
    version="0.1.0",
    description="AI safety research on goal specification for model agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Benjamin Sturgeon",
    author_email="sturgeonkid@gmail.com",
    packages=find_packages(where="src") + ["notebooks", "paul_notebooks"],
    package_dir={
        "": "src",
        "notebooks": "notebooks",
        "paul_notebooks": "paul_notebooks",
    },
    py_modules=[
        "extract_sae_features",
        "impala_dropout",
        "interpretable_impala",
        "perform_sae_analysis",
        "policies_impala",
        "probing",
        "sae",
        "sae_training",
        "third_sae",
        "train_procgen_wandb",
        "visualisation_functions",
    ],
    install_requires=[
        "procgen==0.10.7",
        "ipykernel==6.29.0",
        "stable-baselines3==2.2.1",
        "shimmy==1.3.0",
        "wandb==0.16.3",
        "circrl==1.0.0",
        "seaborn==0.13.2",
        "opencv-python==4.9.0.80",
        "matplotlib",
        "numpy",
        "git+https://github.com/UlisseMini/procgen-tools.git#egg=procgen-tools",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
)
