from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from models.sft_model import SFTModel

class SFTTrainer:
    def __init__(self, model, train_dataset, eval_dataset, tokenizer, config):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.config = config

        self.training_args = TrainingArguments(
            output_dir=config['output_dir'],
            per_device_train_batch_size=config['per_device_train_batch_size'],
            per_device_eval_batch_size=config['per_device_eval_batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            learning_rate=config['learning_rate'],
            logging_steps=config['logging_steps'],
            max_steps=config['max_steps'],
            eval_strategy=config['eval_strategy'],
            eval_steps=config['eval_steps'],
            save_steps=config['save_steps'],
            load_best_model_at_end=config['load_best_model_at_end'],
            report_to=config['report_to'],
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )

    def train(self):
        self.trainer.train()

    def save_model(self, save_path):
        self.trainer.save_model(save_path)