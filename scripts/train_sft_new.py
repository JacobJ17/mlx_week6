#!/usr/bin/env python3
"""
Traditional PyTorch training setup for SFT with explicit training loop, progress bars, and wandb.
"""

import os
# Set environment variables before importing anything else
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm
import logging
import warnings
import wandb

# Suppress warnings
warnings.filterwarnings("ignore", message=".*use_reentrant parameter should be passed explicitly.*")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Wandb configuration
WANDB_PROJECT = "qwen-sft-training"
WANDB_RUN_NAME = "qwen-1.7b-summarization"
USE_WANDB = True  # Set to False to disable wandb

# Model and dataset configuration
MODEL_NAME = "Qwen/Qwen3-1.7B"
DATASET_NAME = "CarperAI/openai_summarize_comparisons"
OUTPUT_DIR = "./qwen-sft-pytorch"

# Data configuration
TRAIN_SUBSET = 10000
EVAL_SUBSET = 2500
MAX_LENGTH = 1024

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training configuration
BATCH_SIZE = 4              
GRADIENT_ACCUMULATION_STEPS = 24  
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
EVAL_EVERY_N_STEPS = 50
SAVE_EVERY_N_STEPS = 200
LOG_EVERY_N_STEPS = 10

# =============================================================================

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_wandb():
    """Initialize wandb if enabled."""
    if not USE_WANDB:
        return
    
    config = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "train_subset": TRAIN_SUBSET,
        "eval_subset": EVAL_SUBSET,
        "max_length": MAX_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
    }
    
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=config
    )
    logger.info("📊 Wandb initialized")

def setup_model_and_tokenizer():
    """Setup model and tokenizer with appropriate configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Model loading configuration
    model_kwargs = {"trust_remote_code": True}

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        logger.info("CUDA not found. Loading model on CPU without quantization.")

    # Load the model and tokenizer
    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    if device == "cpu":
        model.to(device)
    
    return model, tokenizer, device

def setup_lora(model, device):
    """Setup LoRA configuration and apply to model."""
    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    # Log to wandb
    if USE_WANDB:
        wandb.log({
            "trainable_params": trainable_params,
            "total_params": all_params,
            "trainable_percentage": 100 * trainable_params / all_params
        })
    
    return model

def prepare_dataset(tokenizer):
    """Load and preprocess the dataset."""
    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    train_dataset = dataset["train"].select(range(TRAIN_SUBSET))
    eval_dataset = dataset["valid1"].select(range(EVAL_SUBSET))
    
    logger.info(f"Using {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")

    system_prompt = "You are a helpful assistant that summarizes text with the same voice as the author."

    def tokenize_function(examples):
        prompts = []
        for i in range(len(examples["prompt"])):
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize the following post:\n{examples['prompt'][i]}"},
                {"role": "assistant", "content": examples['chosen'][i]}
            ]
            prompts.append(tokenizer.apply_chat_template(chat, tokenize=False))
        
        tokenized_inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        
        return tokenized_inputs

    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
    
    return train_dataset, eval_dataset

def create_dataloaders(train_dataset, eval_dataset, tokenizer):
    """Create PyTorch DataLoaders."""
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=torch.cuda.is_available()
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_dataloader, eval_dataloader

def evaluate_model(model, eval_dataloader, device):
    """Evaluate the model and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        eval_bar = tqdm(eval_dataloader, desc="Evaluating", leave=False)
        for batch in eval_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
            eval_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0

def save_model(model, tokenizer, save_path, is_best=False):
    """Save model and tokenizer."""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save additional info
    save_info = {
        "model_name": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "is_best": is_best
    }
    
    import json
    with open(os.path.join(save_path, "training_info.json"), "w") as f:
        json.dump(save_info, f, indent=2)
    
    logger.info(f"{'🏆 Best model' if is_best else '💾 Model'} saved to: {save_path}")

def train_model(model, tokenizer, train_dataloader, eval_dataloader, device):
    """Main training loop with progress bars and wandb logging."""
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Calculate total steps
    total_steps = NUM_EPOCHS * len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS
    logger.info(f"Total training steps: {total_steps}")
    
    # Training state
    global_step = 0
    best_eval_loss = float('inf')
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Progress bar for the epoch
        epoch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        accumulated_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(epoch_bar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            accumulated_loss += loss.item()
            
            # Update weights every GRADIENT_ACCUMULATION_STEPS
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Update progress bar
                epoch_bar.set_postfix({
                    "loss": f"{accumulated_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "step": global_step
                })
                
                # Log to wandb
                if USE_WANDB:
                    wandb.log({
                        "train/loss": accumulated_loss,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "train/epoch": epoch + (step + 1) / len(train_dataloader),
                        "train/step": global_step
                    })
                
                # Log every N steps
                if global_step % LOG_EVERY_N_STEPS == 0:
                    logger.info(f"Step {global_step}: loss = {accumulated_loss:.4f}")
                
                # Evaluate every N steps
                if global_step % EVAL_EVERY_N_STEPS == 0:
                    eval_loss = evaluate_model(model, eval_dataloader, device)
                    logger.info(f"Step {global_step}: eval_loss = {eval_loss:.4f}")
                    
                    # Log to wandb
                    if USE_WANDB:
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/step": global_step
                        })
                    
                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_model_path = os.path.join(OUTPUT_DIR, "best_model")
                        save_model(model, tokenizer, best_model_path, is_best=True)
                        
                        if USE_WANDB:
                            wandb.log({"eval/best_loss": best_eval_loss})
                
                # Save checkpoint every N steps
                if global_step % SAVE_EVERY_N_STEPS == 0:
                    checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    save_model(model, tokenizer, checkpoint_path)
                
                accumulated_loss = 0
    
    # Final save
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    save_model(model, tokenizer, final_model_path)
    
    # Final evaluation
    final_eval_loss = evaluate_model(model, eval_dataloader, device)
    logger.info(f"Final evaluation loss: {final_eval_loss:.4f}")
    
    if USE_WANDB:
        wandb.log({"eval/final_loss": final_eval_loss})
        wandb.finish()
    
    return model

def main():
    logger.info("🚀 Starting PyTorch SFT training...")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Training samples: {TRAIN_SUBSET}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Wandb enabled: {USE_WANDB}")
    logger.info("-" * 50)
    
    # Setup wandb
    setup_wandb()
    
    # Setup model and tokenizer
    model, tokenizer, device = setup_model_and_tokenizer()
    
    # Setup LoRA
    model = setup_lora(model, device)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(tokenizer)
    
    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, tokenizer)
    
    # Train the model
    model = train_model(model, tokenizer, train_dataloader, eval_dataloader, device)
    
    logger.info("✅ Training completed successfully!")
    logger.info(f"📁 Models saved in: {OUTPUT_DIR}")
    logger.info(f"🏆 Best model: {OUTPUT_DIR}/best_model")
    logger.info(f"📝 Final model: {OUTPUT_DIR}/final_model")

if __name__ == "__main__":
    main()