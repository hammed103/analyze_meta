#!/usr/bin/env python3
"""
Quick image downloader - Download first 20 images as a test
"""

import pandas as pd
import requests
import os
from urllib.parse import urlparse
import time
from PIL import Image
import io

def download_sample_images():
    """Download first 20 images as a test"""
    
    # Create directory
    os.makedirs("sample_images", exist_ok=True)
    
    # Load dataset
    print("📊 Loading dataset...")
    try:
        df = pd.read_csv('Data/facebook_ads_electric_vehicles_with_openai_summaries_cached.csv')
        print(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Filter for rows with image URLs
    df_with_images = df[df['new_image_url'].notna() & (df['new_image_url'] != '')].copy()
    print(f"🖼️  Found {len(df_with_images)} rows with image URLs")
    
    if len(df_with_images) == 0:
        print("❌ No image URLs found in dataset")
        return
    
    # Take first 20 images
    sample_df = df_with_images.head(20)
    print(f"🎯 Downloading first {len(sample_df)} images...")
    
    successful = 0
    failed = 0
    
    for idx, row in sample_df.iterrows():
        ad_id = row['ad_archive_id']
        image_url = row['new_image_url']
        
        # Generate filename
        filename = f"ad_{ad_id}_{idx}.jpg"
        filepath = os.path.join("sample_images", filename)
        
        print(f"📥 {idx+1}/20: Downloading {filename}")
        
        try:
            # Download image
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(image_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Verify it's a valid image
            try:
                with Image.open(filepath) as img:
                    print(f"✅ Success: {img.size} pixels")
                    successful += 1
            except:
                os.remove(filepath)
                print(f"❌ Invalid image format")
                failed += 1
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            failed += 1
        
        # Small delay
        time.sleep(0.5)
    
    print(f"\n📊 Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Images saved to: sample_images/")

if __name__ == "__main__":
    download_sample_images()
