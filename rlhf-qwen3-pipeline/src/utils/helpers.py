def log_metrics(metrics, step):
    """
    Logs the given metrics at the specified training step.
    
    Args:
        metrics (dict): A dictionary containing metric names and their values.
        step (int): The current training step.
    """
    for key, value in metrics.items():
        print(f"Step {step}: {key} = {value}")

def save_model(model, output_dir):
    """
    Saves the trained model to the specified directory.
    
    Args:
        model: The model to be saved.
        output_dir (str): The directory where the model will be saved.
    """
    model.save_pretrained(output_dir)

def load_model(model_class, model_dir):
    """
    Loads a model from the specified directory.
    
    Args:
        model_class: The class of the model to be loaded.
        model_dir (str): The directory from which to load the model.
    
    Returns:
        The loaded model.
    """
    return model_class.from_pretrained(model_dir)

def calculate_average_reward(rewards):
    """
    Calculates the average reward from a list of rewards.
    
    Args:
        rewards (list): A list of reward values.
    
    Returns:
        float: The average reward.
    """
    return sum(rewards) / len(rewards) if rewards else 0.0