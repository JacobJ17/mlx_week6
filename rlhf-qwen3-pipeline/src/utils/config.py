from pathlib import Path
import yaml

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def save_config(self):
        with open(self.config_file, 'w') as file:
            yaml.dump(self.config, file)

# Example usage
if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / 'configs' / 'sft_config.yaml'
    config = Config(config_path)
    print(config.get('learning_rate', 1e-4))