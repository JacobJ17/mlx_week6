from src.utils.config import load_config
from src.data.dataset_loader import load_datasets
from src.data.preprocessing import preprocess_data
from src.models.sft_model import SFTModel
from src.models.reward_model import RewardModel
from src.models.policy_model import PolicyModel
from src.training.sft_trainer import SFTTrainer
from src.training.reward_trainer import RewardTrainer
from src.training.ppo_trainer import PPOTrainer

def main():
    # Load configuration
    sft_config = load_config('configs/sft_config.yaml')
    reward_config = load_config('configs/reward_config.yaml')
    ppo_config = load_config('configs/ppo_config.yaml')

    # Load and preprocess datasets
    train_dataset, eval_dataset = load_datasets(sft_config['dataset'])
    train_dataset = preprocess_data(train_dataset)
    eval_dataset = preprocess_data(eval_dataset)

    # Train the supervised fine-tuning model
    sft_model = SFTModel(sft_config)
    sft_trainer = SFTTrainer(sft_model, train_dataset, eval_dataset, sft_config)
    sft_trainer.train()

    # Train the reward model
    reward_model = RewardModel(reward_config)
    reward_trainer = RewardTrainer(reward_model, train_dataset, reward_config)
    reward_trainer.train()

    # Train the policy model using PPO
    policy_model = PolicyModel(ppo_config)
    ppo_trainer = PPOTrainer(policy_model, train_dataset, reward_model, ppo_config)
    ppo_trainer.train()

if __name__ == "__main__":
    main()