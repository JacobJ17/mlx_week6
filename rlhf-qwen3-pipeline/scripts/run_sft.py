from src.data.dataset_loader import load_dataset
from src.data.preprocessing import preprocess_data
from src.models.sft_model import SFTModel
from src.training.sft_trainer import SFTTrainer
import yaml

def main():
    # Load configuration for SFT
    with open("configs/sft_config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Load and preprocess the dataset
    dataset = load_dataset(config['dataset_name'])
    train_data, eval_data = preprocess_data(dataset, config['preprocessing'])

    # Initialize the SFT model
    model = SFTModel(config['model_params'])

    # Initialize the SFT trainer
    trainer = SFTTrainer(model, train_data, eval_data, config['training_params'])

    # Start the training process
    trainer.train()

if __name__ == "__main__":
    main()