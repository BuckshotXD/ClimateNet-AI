import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List
import time

class ClimateTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str = 'cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total_pixels = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total_pixels += target.numel()
        
        accuracy = 100. * correct / total_pixels
        avg_loss = total_loss / len(self.val_loader)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, epochs: int, save_path: str = 'best_model.pth'):
        best_accuracy = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_metrics["loss"]:.4f}')
            print(f'Val Accuracy: {val_metrics["accuracy"]:.2f}%')
            print(f'Time: {epoch_time:.2f}s')
            print('-' * 50)
            
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': best_accuracy
                }, save_path)