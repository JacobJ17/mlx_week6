class RewardModel:
    def __init__(self, model_name, tokenizer):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = self.load_model()

    def load_model(self):
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def train(self, train_dataset, eval_dataset, training_args):
        from transformers import Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()