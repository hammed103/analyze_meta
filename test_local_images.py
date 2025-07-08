#!/usr/bin/env python3
"""
Test script to verify local image loading functionality
"""

import pandas as pd
import os
import hashlib
from PIL import Image

def find_local_image(ad_id, image_url):
    """Find locally saved image for an ad (same function as in dashboard)"""
    
    # Generate the same filename used in download scripts
    url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
    filename = f"{ad_id}_{url_hash}.jpg"
    
    # Check possible local directories
    local_dirs = [
        "ev_ad_images/by_car_model",
        "ev_ad_images/thumbnails", 
        "sample_images",
        "downloaded_images/originals"
    ]
    
    for base_dir in local_dirs:
        if os.path.exists(base_dir):
            # Search in subdirectories
            for root, dirs, files in os.walk(base_dir):
                if filename in files:
                    return os.path.join(root, filename)
            
            # Also check direct filename match
            direct_path = os.path.join(base_dir, filename)
            if os.path.exists(direct_path):
                return direct_path
    
    return None

def test_local_image_loading():
    """Test local image loading functionality"""
    
    print("🧪 Testing Local Image Loading")
    print("=" * 50)
    
    # Load dataset
    try:
        df = pd.read_csv('Data/facebook_ads_electric_vehicles_with_openai_summaries_cached.csv')
        print(f"✅ Loaded dataset: {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Filter for rows with image URLs
    df_with_images = df[df['new_image_url'].notna() & (df['new_image_url'] != '')].copy()
    print(f"🖼️  Found {len(df_with_images)} rows with image URLs")
    
    if len(df_with_images) == 0:
        print("❌ No image URLs found")
        return
    
    # Check local image directories
    print(f"\n📁 Checking local image directories:")
    local_dirs = [
        "ev_ad_images/by_car_model",
        "ev_ad_images/thumbnails", 
        "sample_images",
        "downloaded_images/originals"
    ]
    
    found_dirs = []
    for dir_path in local_dirs:
        if os.path.exists(dir_path):
            file_count = sum(1 for root, dirs, files in os.walk(dir_path) 
                           for file in files if file.endswith(('.jpg', '.png', '.gif')))
            print(f"   ✅ {dir_path}: {file_count} image files")
            found_dirs.append(dir_path)
        else:
            print(f"   ❌ {dir_path}: Not found")
    
    if not found_dirs:
        print(f"\n⚠️  No local image directories found!")
        print(f"💡 Run one of these to download images:")
        print(f"   python3 quick_image_download.py")
        print(f"   python3 download_all_ev_images.py")
        return
    
    # Test local image loading for first 10 ads
    print(f"\n🔍 Testing local image loading for first 10 ads:")
    test_sample = df_with_images.head(10)
    
    local_found = 0
    local_loadable = 0
    
    for idx, row in test_sample.iterrows():
        ad_id = row['ad_archive_id']
        image_url = row['new_image_url']
        
        # Try to find local image
        local_path = find_local_image(ad_id, image_url)
        
        if local_path:
            local_found += 1
            print(f"   ✅ Ad {ad_id}: Found at {local_path}")
            
            # Try to load the image
            try:
                with Image.open(local_path) as img:
                    print(f"      📏 Size: {img.size}, Mode: {img.mode}")
                    local_loadable += 1
            except Exception as e:
                print(f"      ❌ Cannot load: {e}")
        else:
            print(f"   ❌ Ad {ad_id}: Not found locally")
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"   🎯 Tested: {len(test_sample)} ads")
    print(f"   💾 Found locally: {local_found}")
    print(f"   ✅ Successfully loadable: {local_loadable}")
    print(f"   📈 Local availability: {(local_found/len(test_sample))*100:.1f}%")
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    total_local = 0
    for _, row in df_with_images.iterrows():
        if find_local_image(row['ad_archive_id'], row['new_image_url']):
            total_local += 1
    
    print(f"   🖼️  Total images in dataset: {len(df_with_images)}")
    print(f"   💾 Available locally: {total_local}")
    print(f"   🌐 Need to download: {len(df_with_images) - total_local}")
    print(f"   📊 Local coverage: {(total_local/len(df_with_images))*100:.1f}%")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if total_local == 0:
        print(f"   🚀 Download images: python3 download_all_ev_images.py")
        print(f"   🧪 Or test with: python3 quick_image_download.py")
    elif total_local < len(df_with_images):
        print(f"   📥 Download remaining {len(df_with_images) - total_local} images")
        print(f"   🚀 Run: python3 download_all_ev_images.py")
    else:
        print(f"   ✅ All images available locally!")
        print(f"   🎉 Dashboard will load images quickly from local files")

if __name__ == "__main__":
    test_local_image_loading()
