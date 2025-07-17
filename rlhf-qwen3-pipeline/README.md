# Reinforcement Learning from Human Feedback (RLHF) Pipeline for Qwen3

This project implements a reinforcement learning from human feedback (RLHF) training pipeline using the Qwen3 model. The pipeline consists of three main components: supervised fine-tuning (SFT), reward modeling, and policy optimization. 

## Project Structure

- **src/**: Contains the source code for data handling, model definitions, training processes, and utility functions.
  - **data/**: Includes modules for loading and preprocessing datasets.
  - **models/**: Contains implementations of the SFT model, reward model, and policy model.
  - **training/**: Implements the training logic for each model type.
  - **utils/**: Provides configuration settings and helper functions.
  - **main.py**: The entry point for orchestrating the training processes.

- **configs/**: Contains YAML configuration files for SFT, reward model, and PPO training.

- **scripts/**: Includes scripts to run the training processes for SFT, reward model, and policy model.

- **notebooks/**: Jupyter notebooks for data exploration and training implementations.

- **tests/**: Unit tests for data handling, model implementations, and training processes.

- **requirements.txt**: Lists the dependencies required for the project.

- **setup.py**: Used for packaging the project and managing dependencies.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd rlhf-qwen3-pipeline
pip install -r requirements.txt
```

## Usage

1. **Supervised Fine-Tuning (SFT)**:
   - Run the SFT training script:
     ```bash
     python scripts/run_sft.py
     ```

2. **Reward Model Training**:
   - Train the reward model using:
     ```bash
     python scripts/run_reward_training.py
     ```

3. **Policy Model Training**:
   - Optimize the policy model with PPO:
     ```bash
     python scripts/run_ppo.py
     ```

## Notebooks

Explore the Jupyter notebooks for detailed implementations and visualizations:
- **Data Exploration**: `notebooks/data_exploration.ipynb`
- **SFT Training**: `notebooks/sft_training.ipynb`
- **Reward Training**: `notebooks/reward_training.ipynb`
- **PPO Training**: `notebooks/ppo_training.ipynb`

## Testing

Run the unit tests to ensure the functionality of the project:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for details.