from setuptools import setup, find_packages

setup(
    name="rlhf-qwen3-pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A reinforcement learning from human feedback (RLHF) pipeline for training models based on the Qwen3 architecture.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "datasets>=1.0.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "pyyaml",
        "matplotlib",
        "seaborn",
    ],
    entry_points={
        "console_scripts": [
            "run_sft=scripts.run_sft:main",
            "run_reward_training=scripts.run_reward_training:main",
            "run_ppo=scripts.run_ppo:main",
        ],
    },
)