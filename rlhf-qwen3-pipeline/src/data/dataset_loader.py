from datasets import load_dataset

def load_data(dataset_name, split='train'):
    """
    Load the specified dataset from Hugging Face.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (default is 'train').

    Returns:
        Dataset: The loaded dataset.
    """
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def load_sft_data():
    """
    Load the dataset specifically for supervised fine-tuning.

    Returns:
        Dataset: The loaded dataset for SFT.
    """
    return load_data("CarperAI/openai_summarize_comparisons", split='train')

def load_reward_data():
    """
    Load the dataset specifically for training the reward model.

    Returns:
        Dataset: The loaded dataset for reward training.
    """
    return load_data("CarperAI/openai_summarize_comparisons", split='valid1')