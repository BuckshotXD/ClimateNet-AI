import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple

class Visualizer:
    def __init__(self, class_colors: List[Tuple] = None):
        if class_colors is None:
            self.class_colors = [
                (0, 0, 0),        
                (34, 139, 34),    
                (255, 0, 0),      
                (0, 191, 255),    
                (0, 0, 255)       
            ]
        else:
            self.class_colors = class_colors
    
    def plot_training_history(self, train_losses: List[float], val_losses: List[float], 
                            val_accuracies: List[float], save_path: str = None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        
        ax2.plot(val_accuracies, label='Val Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.set_title('Validation Accuracy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_prediction(self, image: np.ndarray, ground_truth: np.ndarray, 
                           prediction: np.ndarray, save_path: str = None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image[:, :, :3])
        axes[0].set_title('Input Image (RGB)')
        axes[0].axis('off')
        
        axes[1].imshow(ground_truth, cmap='viridis')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(prediction, cmap='viridis')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_change_map(self, prediction1: np.ndarray, prediction2: np.ndarray, 
                         save_path: str = None) -> np.ndarray:
        change_map = np.zeros_like(prediction1)
        
        change_map[(prediction1 != prediction2) & (prediction1 != 0) & (prediction2 != 0)] = 1
        change_map[(prediction1 == 0) & (prediction2 != 0)] = 2
        change_map[(prediction1 != 0) & (prediction2 == 0)] = 3
        
        plt.figure(figsize=(10, 8))
        plt.imshow(change_map, cmap='jet')
        plt.colorbar(label='Change Type')
        plt.title('Environmental Change Detection Map')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return change_map