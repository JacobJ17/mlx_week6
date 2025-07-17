from src.models.reward_model import RewardModel
from src.training.reward_trainer import RewardTrainer
from src.utils.config import load_config
import torch

def main():
    # Load configuration for reward training
    config = load_config('configs/reward_config.yaml')

    # Initialize the reward model
    reward_model = RewardModel(config)

    # Check if a GPU is available and move the model to the appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_model.to(device)

    # Initialize the reward trainer
    reward_trainer = RewardTrainer(reward_model, config)

    # Start the training process
    reward_trainer.train()

if __name__ == "__main__":
    main()