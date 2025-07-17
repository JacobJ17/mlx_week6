#!/usr/bin/env python3
"""
Reward model training for RLHF pipeline.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import logging
import wandb
import warnings

warnings.filterwarnings("ignore", message=".*use_reentrant parameter should be passed explicitly.*")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Wandb configuration
WANDB_PROJECT = "qwen-reward-training"
WANDB_RUN_NAME = "qwen-reward-model"
USE_WANDB = True

# Model configuration
BASE_MODEL_NAME = "Qwen/Qwen3-1.7B"  # Use base model or your SFT model path
SFT_MODEL_PATH = "./qwen-sft-pytorch/best_model"  # Path to your trained SFT model
USE_SFT_MODEL = True  # Set to True to use your SFT model as base

DATASET_NAME = "CarperAI/openai_summarize_comparisons"
OUTPUT_DIR = "./qwen-reward-model"

# Data configuration
TRAIN_SUBSET = 10000
EVAL_SUBSET = 2500
MAX_LENGTH = 1024

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training configuration
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-5  # Lower LR for reward model
NUM_EPOCHS = 1
EVAL_EVERY_N_STEPS = 100
SAVE_EVERY_N_STEPS = 500

# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardModel(nn.Module):
    """Custom reward model that adds a regression head to a causal LM."""
    
    def __init__(self, base_model_path, device="cuda"):
        super().__init__()
        # Load the SFT model as base
        model_kwargs = {"trust_remote_code": True}
        
        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
        
        # Add reward head
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        
        # Initialize reward head
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
        
        if device == "cpu":
            self.to(device)
    
    def forward(self, input_ids, attention_mask=None):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        
        # Use last non-padded token for each sequence
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            last_token_hidden = last_hidden_state[range(batch_size), sequence_lengths]
        else:
            last_token_hidden = last_hidden_state[:, -1]
        
        # Apply reward head
        reward = self.reward_head(last_token_hidden)
        
        # Return in format expected by training loop
        return type('RewardOutput', (), {'logits': reward})()

def setup_wandb():
    """Initialize wandb for reward model training."""
    if not USE_WANDB:
        return
    
    config = {
        "base_model": BASE_MODEL_NAME,
        "sft_model_path": SFT_MODEL_PATH if USE_SFT_MODEL else None,
        "dataset_name": DATASET_NAME,
        "train_subset": TRAIN_SUBSET,
        "eval_subset": EVAL_SUBSET,
        "max_length": MAX_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
    }
    
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config=config)
    logger.info("📊 Wandb initialized for reward model training")

def setup_reward_model():
    """Setup reward model for training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if USE_SFT_MODEL and os.path.exists(SFT_MODEL_PATH):
        logger.info(f"Loading SFT model from: {SFT_MODEL_PATH}")
        model = RewardModel(SFT_MODEL_PATH, device)
        tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    else:
        logger.info(f"Loading base model: {BASE_MODEL_NAME}")
        model = RewardModel(BASE_MODEL_NAME, device)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model.base_model.config, 'pad_token_id'):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer, device

def setup_lora_reward(model, device):
    """Setup LoRA for reward model."""
    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",  # Still causal LM since we're using the base model
    )

    # Apply LoRA only to the base model, not the reward head
    model.base_model = get_peft_model(model.base_model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Reward model trainable parameters: {trainable_params:,} / {all_params:,}")
    
    return model

def prepare_reward_dataset(tokenizer):
    """Prepare dataset for reward model training."""
    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    
    # Skip first 10k samples used for SFT to avoid data leakage
    SFT_TRAIN_SIZE = 10000
    train_start = SFT_TRAIN_SIZE
    train_end = SFT_TRAIN_SIZE + TRAIN_SUBSET
    
    logger.info(f"Skipping first {SFT_TRAIN_SIZE} samples used for SFT")
    logger.info(f"Using training samples {train_start}-{train_end-1}")
    
    train_dataset = dataset["train"].select(range(train_start, min(train_end, len(dataset["train"]))))
    eval_dataset = dataset["valid1"].select(range(EVAL_SUBSET))
    
    # Update the tokenize_pairs function to use chat template:

    def tokenize_pairs(examples):
        """Create chosen/rejected pairs for reward training."""
        chosen_texts = []
        rejected_texts = []
        
        # Use same system prompt as SFT
        system_prompt = "You are a helpful assistant that summarizes text with the same voice as the author."
        
        for i in range(len(examples["prompt"])):
            prompt = examples["prompt"][i]
            chosen = examples["chosen"][i]
            rejected = examples["rejected"][i]
            
            # Create chat format (same as SFT)
            chosen_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize the following post:\n{prompt}"},
                {"role": "assistant", "content": chosen}
            ]
            
            rejected_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize the following post:\n{prompt}"},
                {"role": "assistant", "content": rejected}
            ]
            
            # Apply chat template
            chosen_text = tokenizer.apply_chat_template(chosen_chat, tokenize=False)
            rejected_text = tokenizer.apply_chat_template(rejected_chat, tokenize=False)
            
            chosen_texts.append(chosen_text)
            rejected_texts.append(rejected_text)
        
        # Tokenize both chosen and rejected
        chosen_encodings = tokenizer(
            chosen_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt"
        )
        
        rejected_encodings = tokenizer(
            rejected_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encodings["input_ids"],
            "chosen_attention_mask": chosen_encodings["attention_mask"],
            "rejected_input_ids": rejected_encodings["input_ids"],
            "rejected_attention_mask": rejected_encodings["attention_mask"],
        }
    
    logger.info("Tokenizing reward dataset...")
    train_dataset = train_dataset.map(
        tokenize_pairs, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        tokenize_pairs, 
        batched=True, 
        remove_columns=eval_dataset.column_names
    )
    
    return train_dataset, eval_dataset

class RewardDataCollator:
    """Custom data collator for reward model training."""
    
    def __call__(self, batch):
        chosen_input_ids = torch.stack([item["chosen_input_ids"] for item in batch])
        chosen_attention_mask = torch.stack([item["chosen_attention_mask"] for item in batch])
        rejected_input_ids = torch.stack([item["rejected_input_ids"] for item in batch])
        rejected_attention_mask = torch.stack([item["rejected_attention_mask"] for item in batch])
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }

def reward_loss(chosen_rewards, rejected_rewards):
    """Compute ranking loss for reward model."""
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()

def train_reward_model(model, train_dataloader, eval_dataloader, device, tokenizer):
    """Train the reward model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    global_step = 0
    best_accuracy = 0.0
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting reward training epoch {epoch + 1}/{NUM_EPOCHS}")
        
        epoch_bar = tqdm(train_dataloader, desc=f"Reward Epoch {epoch + 1}")
        accumulated_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(epoch_bar):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get rewards for chosen and rejected
            chosen_outputs = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"]
            )
            rejected_outputs = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"]
            )
            
            chosen_rewards = chosen_outputs.logits.squeeze()
            rejected_rewards = rejected_outputs.logits.squeeze()
            
            # Compute ranking loss
            loss = reward_loss(chosen_rewards, rejected_rewards)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            accumulated_loss += loss.item()
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Calculate accuracy (how often chosen > rejected)
                accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
                
                epoch_bar.set_postfix({
                    "loss": f"{accumulated_loss:.4f}",
                    "acc": f"{accuracy:.3f}",
                    "step": global_step
                })
                
                if USE_WANDB:
                    wandb.log({
                        "reward/loss": accumulated_loss,
                        "reward/accuracy": accuracy,
                        "reward/step": global_step,
                        "reward/chosen_reward_mean": chosen_rewards.mean().item(),
                        "reward/rejected_reward_mean": rejected_rewards.mean().item(),
                    })
                
                if global_step % SAVE_EVERY_N_STEPS == 0:
                    checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    # Save the custom reward model
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, "reward_model.pt"))
                    # Save tokenizer for convenience
                    tokenizer.save_pretrained(checkpoint_path)
                    
                    logger.info(f"Reward model checkpoint saved: {checkpoint_path}")
                
                accumulated_loss = 0
    
    # Final save
    final_path = os.path.join(OUTPUT_DIR, "final_reward_model")
    os.makedirs(final_path, exist_ok=True)
    
    # Save the custom reward model state dict
    torch.save(model.state_dict(), os.path.join(final_path, "reward_model.pt"))
    
    # Save tokenizer
    tokenizer.save_pretrained(final_path)
    
    # Save model config for later loading
    import json
    config = {
        "model_type": "custom_reward_model",
        "base_model_path": SFT_MODEL_PATH if USE_SFT_MODEL else BASE_MODEL_NAME,
        "hidden_size": model.base_model.config.hidden_size,
        "use_sft_model": USE_SFT_MODEL
    }
    with open(os.path.join(final_path, "reward_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Final reward model saved: {final_path}")
    
    return model

def main():
    logger.info("🎯 Starting reward model training...")
    logger.info(f"Base model: {BASE_MODEL_NAME}")
    logger.info(f"Use SFT model: {USE_SFT_MODEL}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("-" * 50)
    
    setup_wandb()
    
    # Setup reward model
    model, tokenizer, device = setup_reward_model()
    model = setup_lora_reward(model, device)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_reward_dataset(tokenizer)
    
    # Create dataloaders
    data_collator = RewardDataCollator()
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
    
    # Train
    model = train_reward_model(model, train_dataloader, eval_dataloader, device, tokenizer)
    
    logger.info("✅ Reward model training completed!")
    logger.info(f"🎯 Reward model saved in: {OUTPUT_DIR}")
    
    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()