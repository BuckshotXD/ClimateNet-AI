import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.downloader import DataDownloader
import ee

def main():
    downloader = DataDownloader()
    
    try:
        downloader.authenticate_gee()
        print("GEE authentication successful")
    except Exception as e:
        print(f"GEE authentication failed: {e}")
        return
    
    region = ee.Geometry.Rectangle([-74.0, -9.0, -53.0, 5.0])
    
    print("Downloading Sentinel-2 data...")
    sentinel_url = downloader.download_sentinel2(region, '2020-01-01', '2020-12-31')
    print(f"Sentinel-2 URL: {sentinel_url}")
    
    print("Downloading Landsat 8 data...")
    landsat_url = downloader.download_landsat8(region, '2020-01-01', '2020-12-31')
    print(f"Landsat 8 URL: {landsat_url}")
    
    print("Downloading weather data...")
    weather_data = downloader.download_weather_data(-3.7, -73.2, '2020-01-01', '2020-12-31')
    print("Weather data downloaded successfully")

if __name__ == "__main__":
    main()