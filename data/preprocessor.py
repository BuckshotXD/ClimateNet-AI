import numpy as np
import rasterio
from rasterio.transform import from_bounds
from typing import Tuple, Dict
import cv2
import torch
from torch.utils.data import Dataset

class ClimateDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            image = self.transform(image)
        
        image = torch.FloatTensor(image).permute(2, 0, 1)
        mask = torch.LongTensor(mask)
        
        return image, mask

class DataPreprocessor:
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        min_val = np.min(image)
        max_val = np.max(image)
        normalized = (image - min_val) / (max_val - min_val)
        return normalized
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
    
    def augment_data(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        augments = {}
        
        augments['original'] = {'image': image, 'mask': mask}
        augments['flip_h'] = {'image': np.fliplr(image), 'mask': np.fliplr(mask)}
        augments['flip_v'] = {'image': np.flipud(image), 'mask': np.flipud(mask)}
        
        return augments
    
    def prepare_training_data(self, images: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        processed_images = []
        processed_masks = []
        
        for img, mask in zip(images, masks):
            img = self.normalize_image(img)
            img = self.resize_image(img)
            mask = self.resize_image(mask)
            
            augments = self.augment_data(img, mask)
            for aug in augments.values():
                processed_images.append(aug['image'])
                processed_masks.append(aug['mask'])
        
        return np.array(processed_images), np.array(processed_masks)