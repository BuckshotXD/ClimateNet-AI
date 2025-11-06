import os
import requests
import rasterio
import numpy as np
from typing import List, Tuple
import ee
import geemap

class DataDownloader:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def authenticate_gee(self):
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
    
    def download_sentinel2(self, region: ee.Geometry, start_date: str, end_date: str) -> str:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        image = collection.median()
        download_url = image.getDownloadURL({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region': region
        })
        
        return download_url
    
    def download_landsat8(self, region: ee.Geometry, start_date: str, end_date: str) -> str:
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(region) \
            .filterDate(start_date, end_date)
        
        image = collection.median()
        download_url = image.getDownloadURL({
            'scale': 30,
            'crs': 'EPSG:4326',
            'region': region
        })
        
        return download_url
    
    def download_weather_data(self, lat: float, lon: float, start_date: str, end_date: str):
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto"
        }
        
        response = requests.get(base_url, params=params)
        return response.json()