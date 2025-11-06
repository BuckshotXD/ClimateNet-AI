import torch
import torch.nn as nn
import torch.nn.functional as F

class ClimateLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
    
    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        return focal_loss
    
    def dice_loss(self, inputs, targets):
        num_classes = inputs.shape[1]
        dice = 0
        
        for class_idx in range(num_classes):
            input_flat = inputs[:, class_idx].contiguous().view(-1)
            target_flat = (targets == class_idx).float().contiguous().view(-1)
            
            intersection = (input_flat * target_flat).sum()
            dice += (2. * intersection + 1e-6) / (input_flat.sum() + target_flat.sum() + 1e-6)
        
        return 1 - (dice / num_classes)
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return focal + dice