from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SFTModel:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, train_dataset, eval_dataset, training_args):
        from transformers import Trainer, TrainingArguments
        training_args = TrainingArguments(
            output_dir=training_args['output_dir'],
            per_device_train_batch_size=training_args['per_device_train_batch_size'],
            per_device_eval_batch_size=training_args['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_args['gradient_accumulation_steps'],
            learning_rate=training_args['learning_rate'],
            logging_steps=training_args['logging_steps'],
            max_steps=training_args['max_steps'],
            eval_strategy=training_args['eval_strategy'],
            eval_steps=training_args['eval_steps'],
            save_steps=training_args['save_steps'],
            load_best_model_at_end=training_args['load_best_model_at_end'],
            report_to=training_args['report_to'],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(training_args['output_dir'])

    def generate(self, input_text: str, max_length: int = 50):
        inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        outputs = self.model.generate(inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)