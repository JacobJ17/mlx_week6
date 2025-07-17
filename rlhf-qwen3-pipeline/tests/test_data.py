from unittest import TestCase
from src.data.dataset_loader import load_dataset
from src.data.preprocessing import preprocess_data

class TestDataFunctions(TestCase):

    def test_load_dataset(self):
        # Test if the dataset loads correctly
        dataset = load_dataset("CarperAI/openai_summarize_comparisons")
        self.assertIsNotNone(dataset)
        self.assertIn("train", dataset)
        self.assertIn("valid1", dataset)

    def test_preprocess_data(self):
        # Test if the preprocessing function works as expected
        raw_data = [{"text": "Sample text for testing."}]
        processed_data = preprocess_data(raw_data)
        self.assertIsInstance(processed_data, list)
        self.assertGreater(len(processed_data), 0)  # Ensure some data is returned
        self.assertIn("input_ids", processed_data[0])  # Check for expected keys in processed data
        self.assertIn("attention_mask", processed_data[0])  # Check for expected keys in processed data