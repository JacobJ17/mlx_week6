from unittest import TestCase
from src.training.sft_trainer import SFTTrainer
from src.training.reward_trainer import RewardTrainer
from src.training.ppo_trainer import PPOTrainer
from src.models.sft_model import SFTModel
from src.models.reward_model import RewardModel
from src.models.policy_model import PolicyModel
from src.utils.config import load_config

class TestTrainingPipelines(TestCase):

    def setUp(self):
        self.sft_model = SFTModel()
        self.reward_model = RewardModel()
        self.policy_model = PolicyModel()
        self.sft_trainer = SFTTrainer(self.sft_model)
        self.reward_trainer = RewardTrainer(self.reward_model)
        self.ppo_trainer = PPOTrainer(self.policy_model)

    def test_sft_training(self):
        config = load_config('configs/sft_config.yaml')
        result = self.sft_trainer.train(config)
        self.assertTrue(result['success'])

    def test_reward_training(self):
        config = load_config('configs/reward_config.yaml')
        result = self.reward_trainer.train(config)
        self.assertTrue(result['success'])

    def test_ppo_training(self):
        config = load_config('configs/ppo_config.yaml')
        result = self.ppo_trainer.train(config)
        self.assertTrue(result['success'])