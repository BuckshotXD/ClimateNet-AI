import torch
import numpy as np
import rasterio
from typing import Dict, Any
import cv2

class ClimatePredictor:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.class_names = {
            0: 'background',
            1: 'deforestation',
            2: 'urban_expansion', 
            3: 'glacier_melt',
            4: 'water_body'
        }
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        checkpoint = torch.load(model_path, map_location=self.device)
        model = ClimateUNet()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32)
        
        min_val = np.min(image, axis=(0, 1), keepdims=True)
        max_val = np.max(image, axis=(0, 1), keepdims=True)
        image = (image - min_val) / (max_val - min_val + 1e-8)
        
        image = cv2.resize(image, (256, 256))
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.to(self.device)
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        class_counts = {}
        for class_id, class_name in self.class_names.items():
            class_counts[class_name] = np.sum(prediction == class_id)
        
        total_pixels = prediction.size
        class_percentages = {name: (count/total_pixels)*100 for name, count in class_counts.items()}
        
        return {
            'prediction_mask': prediction,
            'class_counts': class_counts,
            'class_percentages': class_percentages,
            'dominant_change': max(class_percentages, key=class_percentages.get)
        }
    
    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        predictions = []
        
        for image in images:
            result = self.predict(image)
            predictions.append(result['prediction_mask'])
        
        return np.array(predictions)