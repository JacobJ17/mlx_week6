from transformers import PPOTrainer, TrainingArguments
import torch

class PPOTrainer:
    def __init__(self, policy_model, reward_model, config):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.config = config
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=config['learning_rate'])
        self.training_args = TrainingArguments(
            output_dir=config['output_dir'],
            per_device_train_batch_size=config['batch_size'],
            num_train_epochs=config['num_epochs'],
            logging_dir=config['logging_dir'],
            logging_steps=config['logging_steps'],
        )

    def train(self, train_dataset):
        for epoch in range(self.training_args.num_train_epochs):
            for step, batch in enumerate(train_dataset):
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.policy_model(batch['input_ids'])
                # Compute rewards
                rewards = self.reward_model(outputs)
                # Compute loss
                loss = -torch.mean(rewards)  # Maximize rewards
                loss.backward()
                self.optimizer.step()

                if step % self.training_args.logging_steps == 0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

    def save_model(self, save_path):
        self.policy_model.save_pretrained(save_path)