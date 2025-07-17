#!/usr/bin/env python3
"""
PPO training for RLHF pipeline - fixed data splits and reward model loading.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import wandb
import logging
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# Wandb configuration
WANDB_PROJECT = "qwen-ppo-training"
WANDB_RUN_NAME = "qwen-ppo-rlhf"
USE_WANDB = True

# Model paths
SFT_MODEL_PATH = "./qwen-sft-pytorch/best_model"
REWARD_MODEL_PATH = "./qwen-reward-model/final_reward_model"
OUTPUT_DIR = "./qwen-ppo-final"

# Data configuration - Skip data used in previous stages
DATASET_NAME = "CarperAI/openai_summarize_comparisons"
SFT_TRAIN_SIZE = 10000      # Skip samples 0-9,999 (used for SFT)
REWARD_TRAIN_SIZE = 15000   # Skip samples 10k-24,999 (used for reward model)
PPO_TRAIN_SUBSET = 5000     # Use samples 25k-29,999 for PPO
MAX_LENGTH = 512
MIN_LENGTH = 50

# PPO configuration
PPO_CONFIG = PPOConfig(
    model_name=SFT_MODEL_PATH,
    learning_rate=1e-5,
    batch_size=8,           # Smaller batch for PPO
    mini_batch_size=2,      # Even smaller mini batches
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=0.1,          # KL divergence constraint
    ppo_epochs=4,
    seed=42,
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
    vf_coef=0.1,           # Value function coefficient
    cliprange=0.2,         # PPO clipping range
)

# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardModel(nn.Module):
    """Custom reward model that adds a regression head to a causal LM."""
    
    def __init__(self, base_model_path):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
        
        # Add reward head
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        
        # Initialize reward head
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
    
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
    """Initialize wandb for PPO training."""
    if not USE_WANDB:
        return
    
    config = vars(PPO_CONFIG).copy()
    config.update({
        "sft_model_path": SFT_MODEL_PATH,
        "reward_model_path": REWARD_MODEL_PATH,
        "sft_train_size": SFT_TRAIN_SIZE,
        "reward_train_size": REWARD_TRAIN_SIZE,
        "ppo_train_subset": PPO_TRAIN_SUBSET,
    })
    
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=config
    )
    logger.info("📊 Wandb initialized for PPO training")

def load_models():
    """Load SFT model with value head and reward model."""
    logger.info(f"Loading SFT model from: {SFT_MODEL_PATH}")
    
    # Load SFT model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(SFT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    
    # Load custom reward model
    logger.info(f"Loading reward model from: {REWARD_MODEL_PATH}")
    
    # Load the reward model config to get base model path
    config_path = os.path.join(REWARD_MODEL_PATH, "reward_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            reward_config = json.load(f)
        base_model_path = reward_config.get("base_model_path", SFT_MODEL_PATH)
    else:
        logger.warning("No reward config found, using SFT model path as base")
        base_model_path = SFT_MODEL_PATH
    
    # Create reward model and load weights
    reward_model = RewardModel(base_model_path)
    reward_state_dict = torch.load(
        os.path.join(REWARD_MODEL_PATH, "reward_model.pt"),
        map_location="cpu"
    )
    reward_model.load_state_dict(reward_state_dict)
    
    # Move reward model to same device as main model
    if torch.cuda.is_available():
        reward_model = reward_model.cuda()
    
    reward_model.eval()  # Set to evaluation mode
    
    return model, tokenizer, reward_model

def prepare_ppo_dataset(tokenizer):
    """Prepare dataset for PPO training - skip previously used data."""
    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    
    # Skip data used in SFT and reward model training
    ppo_start = SFT_TRAIN_SIZE + REWARD_TRAIN_SIZE  # Start at 25k
    ppo_end = ppo_start + PPO_TRAIN_SUBSET          # End at 30k
    
    logger.info(f"Skipping first {SFT_TRAIN_SIZE} samples (SFT training)")
    logger.info(f"Skipping next {REWARD_TRAIN_SIZE} samples (reward training)")
    logger.info(f"Using PPO samples {ppo_start}-{ppo_end-1}")
    
    train_dataset = dataset["train"].select(range(ppo_start, min(ppo_end, len(dataset["train"]))))
    logger.info(f"PPO dataset size: {len(train_dataset)}")
    
    # Use same system prompt as SFT/reward training for consistency
    system_prompt = "You are a helpful assistant that summarizes text with the same voice as the author."
    
    def tokenize_function(examples):
        """Tokenize prompts for PPO training using chat template."""
        prompts = []
        for prompt in examples["prompt"]:
            # Create chat format (consistent with SFT training)
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize the following post:\n{prompt}"},
            ]
            # Don't include assistant response - PPO will generate it
            formatted_prompt = tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=True  # Adds the assistant prompt
            )
            prompts.append(formatted_prompt)
        
        return tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_LENGTH // 2,  # Leave room for generation
            padding=False,
        )
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    return train_dataset

def compute_reward(reward_model, query_tensors, response_tensors, tokenizer):
    """Compute rewards using the trained reward model."""
    rewards = []
    
    # Same system prompt as training
    system_prompt = "You are a helpful assistant that summarizes text with the same voice as the author."
    
    for query, response in zip(query_tensors, response_tensors):
        try:
            # Decode the generated response
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            
            # Decode the original query to extract the prompt
            query_text = tokenizer.decode(query, skip_special_tokens=True)
            
            # Extract just the post content from the query
            # This is a bit hacky but necessary to reconstruct the original format
            if "Summarize the following post:" in query_text:
                post_content = query_text.split("Summarize the following post:\n")[1].split("\n")[0]
            else:
                post_content = query_text  # Fallback
            
            # Create the full chat format that reward model expects
            full_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize the following post:\n{post_content}"},
                {"role": "assistant", "content": response_text}
            ]
            
            # Format with chat template (same as reward model training)
            full_text = tokenizer.apply_chat_template(full_chat, tokenize=False)
            
            # Tokenize for reward model
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True
            )
            
            # Move to same device as reward model
            inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
            
            # Get reward score
            with torch.no_grad():
                outputs = reward_model(**inputs)
                reward = outputs.logits.squeeze().cpu().item()
            
            rewards.append(reward)
            
        except Exception as e:
            logger.warning(f"Error computing reward: {e}, using default reward of 0.0")
            rewards.append(0.0)
    
    return torch.tensor(rewards)

def test_final_model(model_path, tokenizer_path, test_prompt="Write a short post about the benefits of exercise"):
    """Test the final PPO model with a sample prompt."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Testing final model from: {model_path}")
        
        # Load the final model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Create test prompt in same format as training
        system_prompt = "You are a helpful assistant that summarizes text with the same voice as the author."
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize the following post:\n{test_prompt}"},
        ]
        
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_summary = full_response[len(prompt):].strip()
        
        logger.info("🧪 Model Test Results:")
        logger.info(f"Prompt: {test_prompt}")
        logger.info(f"Generated Summary: {generated_summary}")
        
        return generated_summary
        
    except Exception as e:
        logger.error(f"Failed to test model: {e}")
        return None

def main():
    logger.info("🚀 Starting PPO training...")
    logger.info(f"SFT model: {SFT_MODEL_PATH}")
    logger.info(f"Reward model: {REWARD_MODEL_PATH}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Data splits - SFT: 0-{SFT_TRAIN_SIZE-1}, Reward: {SFT_TRAIN_SIZE}-{SFT_TRAIN_SIZE+REWARD_TRAIN_SIZE-1}")
    logger.info(f"PPO data: {SFT_TRAIN_SIZE+REWARD_TRAIN_SIZE}-{SFT_TRAIN_SIZE+REWARD_TRAIN_SIZE+PPO_TRAIN_SUBSET-1}")
    logger.info("-" * 50)
    
    setup_wandb()
    
    # Load models
    model, tokenizer, reward_model = load_models()
    
    # Prepare dataset
    dataset = prepare_ppo_dataset(tokenizer)
    
    # Setup PPO trainer
    ppo_trainer = PPOTrainer(
        config=PPO_CONFIG,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    
    # Generation settings for creating TLDR-style summaries
    generation_kwargs = {
        "min_length": MIN_LENGTH,
        "max_new_tokens": 100,     # Keep summaries concise
        "do_sample": True,
        "top_k": 50,               # Top-k sampling for diversity
        "top_p": 0.95,             # Nucleus sampling
        "temperature": 0.7,        # Control randomness
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    logger.info("Starting PPO training loop...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_steps = 0
    
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs
        )
        
        # Compute rewards using trained reward model
        rewards = compute_reward(reward_model, query_tensors, response_tensors, tokenizer)
        
        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        total_steps += 1
        
        # Log statistics
        if USE_WANDB:
            wandb.log({
                **stats,
                "ppo/epoch": epoch,
                "ppo/reward_mean": rewards.mean().item(),
                "ppo/reward_std": rewards.std().item(),
                "ppo/reward_min": rewards.min().item(),
                "ppo/reward_max": rewards.max().item(),
            })
        
        # Print progress with sample generation
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: reward_mean={rewards.mean():.3f}, reward_std={rewards.std():.3f}")
            
            # Log a sample generation
            if len(response_tensors) > 0:
                sample_response = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
                logger.info(f"Sample summary: {sample_response[:100]}...")
        
        # Save checkpoint
        if epoch % 100 == 0 and epoch > 0:
            checkpoint_path = f"{OUTPUT_DIR}/checkpoint-{epoch}"
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save PPO model
            ppo_trainer.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # Save checkpoint metadata
            checkpoint_info = {
                "epoch": epoch,
                "total_steps": total_steps,
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "model_type": "ppo_checkpoint",
                "base_sft_model": SFT_MODEL_PATH,
                "reward_model": REWARD_MODEL_PATH,
            }
            with open(os.path.join(checkpoint_path, "checkpoint_info.json"), "w") as f:
                json.dump(checkpoint_info, f, indent=2)
                
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Stop after reasonable number of steps
        if total_steps >= 500:  # Adjust as needed
            logger.info(f"Stopping after {total_steps} steps")
            break
    
    # Final save with comprehensive metadata
    logger.info("Saving final PPO model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the PPO model
    ppo_trainer.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save comprehensive training info
    final_info = {
        "model_type": "ppo_final",
        "base_sft_model": SFT_MODEL_PATH,
        "reward_model": REWARD_MODEL_PATH,
        "total_epochs": epoch,
        "total_steps": total_steps,
        "final_reward_mean": rewards.mean().item(),
        "final_reward_std": rewards.std().item(),
        "training_config": {
            "learning_rate": PPO_CONFIG.learning_rate,
            "batch_size": PPO_CONFIG.batch_size,
            "mini_batch_size": PPO_CONFIG.mini_batch_size,
            "ppo_epochs": PPO_CONFIG.ppo_epochs,
            "target_kl": PPO_CONFIG.target_kl,
            "cliprange": PPO_CONFIG.cliprange,
        },
        "data_splits": {
            "sft_samples": f"0-{SFT_TRAIN_SIZE-1}",
            "reward_samples": f"{SFT_TRAIN_SIZE}-{SFT_TRAIN_SIZE+REWARD_TRAIN_SIZE-1}",
            "ppo_samples": f"{SFT_TRAIN_SIZE+REWARD_TRAIN_SIZE}-{SFT_TRAIN_SIZE+REWARD_TRAIN_SIZE+PPO_TRAIN_SUBSET-1}",
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump(final_info, f, indent=2)
    
    # Create a simple README for the final model
    readme_content = f"""# RLHF-Trained Qwen Model

This model has been trained using the full RLHF pipeline:

1. **Supervised Fine-Tuning (SFT)**: Base model fine-tuned on chosen summaries
2. **Reward Model Training**: Learned to prefer chosen over rejected summaries  
3. **PPO Training**: Reinforcement learning using the reward model

## Model Details
- Base Model: {SFT_MODEL_PATH}
- Reward Model: {REWARD_MODEL_PATH}
- Final PPO Model: {OUTPUT_DIR}

## Training Stats
- Total Training Steps: {total_steps}
- Final Reward Mean: {rewards.mean().item():.3f}
- Final Reward Std: {rewards.std().item():.3f}

## Usage
Load the model using:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{OUTPUT_DIR}")
tokenizer = AutoTokenizer.from_pretrained("{OUTPUT_DIR}")
```

This model should generate high-quality TLDR-style summaries!
"""
    
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
        f.write(readme_content)
    
    logger.info(f"Final PPO model saved: {OUTPUT_DIR}")
    logger.info(f"📄 Training info saved: {OUTPUT_DIR}/training_info.json")
    logger.info(f"📖 README created: {OUTPUT_DIR}/README.md")
    
    if USE_WANDB:
        wandb.finish()
    
    logger.info("✅ PPO training completed!")
    logger.info("🎯 Your model should now generate better TLDR-style summaries!")
    
    # Test the final model
    logger.info("🧪 Testing final model...")
    test_result = test_final_model(OUTPUT_DIR, OUTPUT_DIR)
    if test_result:
        logger.info("✅ Model test completed successfully!")
    else:
        logger.warning("⚠️ Model test failed, but model should still be saved correctly")

if __name__ == "__main__":
    main()