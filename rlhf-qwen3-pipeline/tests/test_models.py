from unittest import TestCase
from src.models.sft_model import SFTModel
from src.models.reward_model import RewardModel
from src.models.policy_model import PolicyModel

class TestModels(TestCase):

    def setUp(self):
        self.sft_model = SFTModel()
        self.reward_model = RewardModel()
        self.policy_model = PolicyModel()

    def test_sft_model_initialization(self):
        self.assertIsNotNone(self.sft_model)

    def test_reward_model_initialization(self):
        self.assertIsNotNone(self.reward_model)

    def test_policy_model_initialization(self):
        self.assertIsNotNone(self.policy_model)

    def test_sft_model_forward_pass(self):
        input_data = "Sample input for SFT model"
        output = self.sft_model.forward(input_data)
        self.assertIsInstance(output, str)  # Assuming the output is a string

    def test_reward_model_evaluation(self):
        sample_output = "Sample output from policy model"
        reward = self.reward_model.evaluate(sample_output)
        self.assertIsInstance(reward, float)  # Assuming the reward is a float

    def test_policy_model_action_selection(self):
        state = "Sample state for policy model"
        action = self.policy_model.select_action(state)
        self.assertIn(action, self.policy_model.action_space)  # Assuming action_space is defined in PolicyModel

    def tearDown(self):
        del self.sft_model
        del self.reward_model
        del self.policy_model