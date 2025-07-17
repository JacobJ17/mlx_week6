#!/usr/bin/env python3
"""
Complete RLHF training pipeline: SFT -> Reward Model -> PPO
"""

import subprocess
import sys
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors."""
    logger.info(f"🚀 Starting: {description}")
    logger.info(f"Command: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info(f"✅ Completed: {description}")
    else:
        logger.error(f"❌ Failed: {description}")
        logger.error(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result

def main():
    logger.info("🔥 Starting full RLHF pipeline...")
    
    # Step 1: Train SFT model
    if not os.path.exists("./qwen-sft-pytorch/best_model"):
        run_command(
            "scripts/train_sft_new.py",
            "SFT Training"
        )
    else:
        logger.info("✅ SFT model already exists, skipping...")
    
    # Step 2: Train reward model
    if not os.path.exists("./qwen-reward-model/final_reward_model"):
        run_command(
            "scripts/reward.py",
            "Reward Model Training"
        )
    else:
        logger.info("✅ Reward model already exists, skipping...")
    
    # Step 3: Train with PPO
    run_command(
        "scripts/ppo.py",
        "PPO Training"
    )
    
    logger.info("🎉 Full RLHF pipeline completed!")
    logger.info("📁 Final model: ./qwen-ppo-final")

if __name__ == "__main__":
    main()