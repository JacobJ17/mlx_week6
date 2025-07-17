from src.utils.config import load_config
from src.training.ppo_trainer import PPOTrainer
from src.models.policy_model import PolicyModel
from src.data.dataset_loader import load_datasets

def main():
    # Load configuration for PPO training
    config = load_config('configs/ppo_config.yaml')

    # Load datasets
    train_dataset, eval_dataset = load_datasets(config['dataset'])

    # Initialize the policy model
    policy_model = PolicyModel(config['model'])

    # Initialize the PPO trainer
    ppo_trainer = PPOTrainer(
        model=policy_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config['training']
    )

    # Start training
    ppo_trainer.train()

if __name__ == "__main__":
    main()