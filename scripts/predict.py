import numpy as np
import rasterio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.predictor import ClimatePredictor
from utils.visualization import Visualizer

def main():
    predictor = ClimatePredictor('models/best_model.pth')
    visualizer = Visualizer()
    
    image_path = 'data/raw/sample_image.tif'
    
    with rasterio.open(image_path) as src:
        image = src.read()
        image = np.moveaxis(image, 0, -1)
    
    result = predictor.predict(image)
    
    print("Prediction Results:")
    print(f"Dominant Change: {result['dominant_change']}")
    print("Class Percentages:")
    for class_name, percentage in result['class_percentages'].items():
        print(f"  {class_name}: {percentage:.2f}%")
    
    visualizer.visualize_prediction(
        image=image,
        ground_truth=np.zeros((256, 256)),
        prediction=result['prediction_mask'],
        save_path='results/prediction_visualization.png'
    )

if __name__ == "__main__":
    main()