import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessor import ClimateDataset, DataPreprocessor
from models.unet import ClimateUNet
from training.trainer import ClimateTrainer
from training.losses import ClimateLoss
from utils.config import ClimateConfig

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = ClimateConfig()
    
    images = np.load('data/processed/train_images.npy')
    masks = np.load('data/processed/train_masks.npy')
    
    dataset = ClimateDataset(images, masks)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.DATA['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.DATA['batch_size'], shuffle=False)
    
    model = ClimateUNet(
        in_channels=config.MODEL['unet']['in_channels'],
        num_classes=config.MODEL['unet']['num_classes'],
        base_channels=config.MODEL['unet']['base_channels']
    )
    
    criterion = ClimateLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.TRAINING['learning_rate'])
    
    trainer = ClimateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    print("Starting training...")
    trainer.train(epochs=config.TRAINING['epochs'], save_path='models/best_model.pth')
    
    print("Training completed!")

if __name__ == "__main__":
    main()