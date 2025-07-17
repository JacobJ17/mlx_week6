#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for Qwen model on summarization task.
Converted from working sandbox.ipynb notebook.
"""

import torch
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# =============================================================================
# CONFIGURATION - Edit these values to customize your training
# =============================================================================

# Model and dataset configuration
MODEL_NAME = "Qwen/Qwen3-1.7B"
DATASET_NAME = "CarperAI/openai_summarize_comparisons"
OUTPUT_DIR = "./qwen-sft-poc"

# Data configuration
TRAIN_SUBSET = 10000          # Number of training samples to use
EVAL_SUBSET = 2500            # Number of evaluation samples to use
MAX_LENGTH = 1024            # Maximum sequence length

# LoRA configuration
LORA_R = 16                  # LoRA rank
LORA_ALPHA = 32              # LoRA alpha (scaling factor)
LORA_DROPOUT = 0.05          # LoRA dropout rate

# Training configuration
PER_DEVICE_TRAIN_BATCH_SIZE = 12
PER_DEVICE_EVAL_BATCH_SIZE = 12
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
MAX_STEPS = 200              # Total training steps
EVAL_STEPS = 25              # Evaluate every N steps
SAVE_STEPS = 100             # Save checkpoint every N steps
LOGGING_STEPS = 10           # Log every N steps

# =============================================================================

def setup_model_and_tokenizer():
    """Setup model and tokenizer with appropriate configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model loading configuration
    model_kwargs = {"trust_remote_code": True}

    if device == "cuda":
        # Configure quantization for 4-bit model loading on GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        # Load the model in full precision on CPU
        print("CUDA not found. Loading model on CPU without quantization.")

    # Load the model and tokenizer
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Set the padding token to be the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False  # Disable cache for training

    # If on CPU, we need to explicitly move the model to the device
    if device == "cpu":
        model.to(device)
    
    return model, tokenizer, device

def setup_lora(model, device):
    """Setup LoRA configuration and apply to model."""
    # Prepare the model for k-bit training only if on GPU
    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Add LoRA adapters to the model
    model = get_peft_model(model, lora_config)
    print(f"LoRA adapters added. Trainable parameters: {model.num_parameters()}")
    
    return model

def prepare_dataset(tokenizer):
    """Load and preprocess the dataset."""
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    # We only need the 'train' and 'valid1' splits for this SFT example
    train_dataset = dataset["train"]
    eval_dataset = dataset["valid1"]

    # For this PoC, let's use a smaller subset of the data
    train_dataset = train_dataset.select(range(TRAIN_SUBSET))
    eval_dataset = eval_dataset.select(range(EVAL_SUBSET))
    
    print(f"Using {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")

    # System prompt to guide the model
    system_prompt = "You are a helpful assistant that summarizes text with the same voice as the author."

    def tokenize_function(examples):
        """
        Applies the chat template to the prompt and chosen summary,
        then tokenizes the result.
        """
        # Create the full chat prompt for each example in the batch
        prompts = []
        for i in range(len(examples["prompt"])):
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize the following post:\n{examples['prompt'][i]}"},
                {"role": "assistant", "content": examples['chosen'][i]}
            ]
            prompts.append(tokenizer.apply_chat_template(chat, tokenize=False))
        
        # Tokenize the formatted prompts
        tokenized_inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # The data collator will handle padding
        )
        
        return tokenized_inputs

    # Apply the tokenization function to the datasets
    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    return tokenized_train_dataset, tokenized_eval_dataset

def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Train the model using the prepared datasets."""
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        max_steps=MAX_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb or other reporting
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start the training
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    final_checkpoint_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(final_checkpoint_path)
    print(f"Model saved to: {final_checkpoint_path}")
    
    return trainer

def main():
    print("🚀 Starting SFT training...")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Training samples: {TRAIN_SUBSET}")
    print(f"Max steps: {MAX_STEPS}")
    print("-" * 50)
    
    # Setup model and tokenizer
    model, tokenizer, device = setup_model_and_tokenizer()
    
    # Setup LoRA
    model = setup_lora(model, device)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(tokenizer)
    
    # Train the model
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset)
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()