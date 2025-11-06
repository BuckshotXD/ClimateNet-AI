import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            value = value.get(k, {})
            if value == {}:
                return default
        
        return value
    
    def update(self, key: str, value: Any):
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            config_ref = config_ref.setdefault(k, {})
        
        config_ref[keys[-1]] = value

class ClimateConfig:
    DATA = {
        'image_size': [256, 256],
        'num_bands': 13,
        'batch_size': 8,
        'validation_split': 0.2
    }
    
    MODEL = {
        'unet': {
            'in_channels': 13,
            'num_classes': 5,
            'base_channels': 64
        },
        'transformer': {
            'input_dim': 13,
            'num_classes': 5,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6
        }
    }
    
    TRAINING = {
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 10
    }