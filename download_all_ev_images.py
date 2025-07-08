#!/usr/bin/env python3
"""
Download all EV ad images from new_image_url column
Organized by car model and with metadata
"""

import pandas as pd
import requests
import os
from urllib.parse import urlparse
import time
from PIL import Image
import json
import hashlib
from datetime import datetime

def create_organized_directories():
    """Create organized directory structure"""
    base_dir = "ev_ad_images"
    subdirs = [
        "by_car_model",
        "by_advertiser", 
        "thumbnails",
        "metadata",
        "failed"
    ]
    
    for subdir in [base_dir] + [os.path.join(base_dir, sub) for sub in subdirs]:
        os.makedirs(subdir, exist_ok=True)
    
    print(f"📁 Created directory structure in: {base_dir}/")
    return base_dir

def safe_filename(text, max_length=50):
    """Create safe filename from text"""
    if pd.isna(text) or text == '':
        return "unknown"
    
    # Remove special characters and limit length
    safe = "".join(c for c in str(text) if c.isalnum() or c in (' ', '-', '_')).strip()
    safe = safe.replace(' ', '_')[:max_length]
    return safe if safe else "unknown"

def download_image_with_metadata(url, filepath, metadata, timeout=20):
    """Download image and save metadata"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Save original image
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify and create thumbnail
        try:
            with Image.open(filepath) as img:
                # Update metadata with image info
                metadata['image_size'] = img.size
                metadata['image_mode'] = img.mode
                metadata['file_size'] = os.path.getsize(filepath)
                
                # Create thumbnail
                thumbnail_dir = filepath.replace('by_car_model', 'thumbnails').replace('by_advertiser', 'thumbnails')
                os.makedirs(os.path.dirname(thumbnail_dir), exist_ok=True)
                
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                img.save(thumbnail_dir, optimize=True, quality=85)
                
                return True, "Success"
        except Exception as e:
            os.remove(filepath)
            return False, f"Invalid image: {e}"
            
    except Exception as e:
        return False, f"Download failed: {e}"

def process_all_images():
    """Download and organize all images"""
    
    # Load dataset
    print("📊 Loading dataset...")
    try:
        df = pd.read_csv('Data/facebook_ads_electric_vehicles_with_openai_summaries_cached.csv')
        print(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Filter for images
    df_images = df[df['new_image_url'].notna() & (df['new_image_url'] != '')].copy()
    print(f"🖼️  Found {len(df_images)} images to download")
    
    if len(df_images) == 0:
        print("❌ No images found")
        return
    
    # Create directories
    base_dir = create_organized_directories()
    
    # Statistics
    stats = {
        'total_images': len(df_images),
        'successful': 0,
        'failed': 0,
        'by_car_model': {},
        'by_advertiser': {},
        'start_time': datetime.now().isoformat(),
        'failed_downloads': []
    }
    
    print(f"\n🚀 Starting download of {len(df_images)} images...")
    
    for idx, row in df_images.iterrows():
        ad_id = row['ad_archive_id']
        image_url = row['new_image_url']
        car_model = safe_filename(row.get('matched_car_models', 'unknown'))
        advertiser = safe_filename(row.get('advertiser_name', 'unknown'))
        
        # Create metadata
        metadata = {
            'ad_id': ad_id,
            'image_url': image_url,
            'car_model': row.get('matched_car_models', ''),
            'advertiser_name': row.get('advertiser_name', ''),
            'page_name': row.get('page_name', ''),
            'ad_title': row.get('ad_title', ''),
            'start_date': row.get('start_date', ''),
            'end_date': row.get('end_date', ''),
            'download_date': datetime.now().isoformat()
        }
        
        # Generate filename
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
        filename = f"{ad_id}_{url_hash}.jpg"
        
        # Organize by car model
        model_dir = os.path.join(base_dir, 'by_car_model', car_model)
        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, filename)
        
        # Also organize by advertiser
        advertiser_dir = os.path.join(base_dir, 'by_advertiser', advertiser)
        os.makedirs(advertiser_dir, exist_ok=True)
        advertiser_filepath = os.path.join(advertiser_dir, filename)
        
        print(f"📥 {stats['successful'] + stats['failed'] + 1}/{len(df_images)}: {car_model}/{filename}")
        
        # Download to car model directory
        success, message = download_image_with_metadata(image_url, filepath, metadata)
        
        if success:
            # Copy to advertiser directory (create symlink or copy)
            try:
                import shutil
                shutil.copy2(filepath, advertiser_filepath)
            except:
                pass  # Skip if copy fails
            
            # Save metadata
            metadata_path = os.path.join(base_dir, 'metadata', f"{ad_id}_{url_hash}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            stats['successful'] += 1
            stats['by_car_model'][car_model] = stats['by_car_model'].get(car_model, 0) + 1
            stats['by_advertiser'][advertiser] = stats['by_advertiser'].get(advertiser, 0) + 1
            
            print(f"✅ Success: {metadata.get('image_size', 'Unknown size')}")
            
        else:
            stats['failed'] += 1
            stats['failed_downloads'].append({
                'ad_id': ad_id,
                'url': image_url,
                'error': message,
                'car_model': car_model,
                'advertiser': advertiser
            })
            print(f"❌ Failed: {message}")
        
        # Progress update every 50 images
        if (stats['successful'] + stats['failed']) % 50 == 0:
            print(f"\n📊 Progress Update:")
            print(f"   ✅ Successful: {stats['successful']}")
            print(f"   ❌ Failed: {stats['failed']}")
            print(f"   📈 Success rate: {(stats['successful']/(stats['successful']+stats['failed']))*100:.1f}%")
            print("-" * 50)
        
        # Small delay to be respectful
        time.sleep(0.5)
    
    # Final statistics
    stats['end_time'] = datetime.now().isoformat()
    stats['success_rate'] = (stats['successful'] / (stats['successful'] + stats['failed'])) * 100
    
    # Save final stats
    stats_path = os.path.join(base_dir, 'download_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save failed downloads
    if stats['failed_downloads']:
        failed_df = pd.DataFrame(stats['failed_downloads'])
        failed_path = os.path.join(base_dir, 'failed', 'failed_downloads.csv')
        failed_df.to_csv(failed_path, index=False)
    
    # Print final summary
    print(f"\n" + "="*60)
    print(f"📊 FINAL DOWNLOAD SUMMARY")
    print(f"="*60)
    print(f"🎯 Total images: {stats['total_images']}")
    print(f"✅ Successful: {stats['successful']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📈 Success rate: {stats['success_rate']:.1f}%")
    
    print(f"\n📁 Directory structure created:")
    print(f"   📂 {base_dir}/by_car_model/ - Images organized by car model")
    print(f"   📂 {base_dir}/by_advertiser/ - Images organized by advertiser")
    print(f"   📂 {base_dir}/thumbnails/ - 200x200 thumbnails")
    print(f"   📂 {base_dir}/metadata/ - JSON metadata for each image")
    
    print(f"\n🏆 Top car models by image count:")
    for model, count in sorted(stats['by_car_model'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {model}: {count} images")
    
    print(f"\n📊 Statistics saved to: {stats_path}")

if __name__ == "__main__":
    process_all_images()
