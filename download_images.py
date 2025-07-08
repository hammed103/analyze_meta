#!/usr/bin/env python3
"""
Download and save images locally from the new_image_url column
"""

import pandas as pd
import requests
import os
from urllib.parse import urlparse
import time
from PIL import Image
import io
import hashlib

def create_image_directory():
    """Create directory structure for storing images"""
    base_dir = "downloaded_images"
    subdirs = ["originals", "thumbnails", "failed"]
    
    for subdir in [base_dir] + [os.path.join(base_dir, sub) for sub in subdirs]:
        os.makedirs(subdir, exist_ok=True)
        print(f"📁 Created directory: {subdir}")
    
    return base_dir

def get_file_extension_from_url(url):
    """Extract file extension from URL"""
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    # Common image extensions
    if '.jpg' in path or '.jpeg' in path:
        return '.jpg'
    elif '.png' in path:
        return '.png'
    elif '.gif' in path:
        return '.gif'
    elif '.webp' in path:
        return '.webp'
    else:
        return '.jpg'  # Default to jpg

def generate_filename(ad_id, url):
    """Generate a unique filename for the image"""
    # Create a hash of the URL for uniqueness
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    extension = get_file_extension_from_url(url)
    
    return f"ad_{ad_id}_{url_hash}{extension}"

def download_image(url, filepath, timeout=30):
    """Download a single image with error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check if it's actually an image
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'gif', 'webp']):
            return False, f"Not an image: {content_type}"
        
        # Save the image
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify the image can be opened
        try:
            with Image.open(filepath) as img:
                # Create thumbnail
                thumbnail_path = filepath.replace('originals', 'thumbnails')
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                img.save(thumbnail_path, optimize=True, quality=85)
                
                return True, f"Downloaded and thumbnailed: {img.size}"
        except Exception as e:
            os.remove(filepath)  # Remove corrupted file
            return False, f"Corrupted image: {e}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Download error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def load_dataset():
    """Load the dataset with image URLs"""
    try:
        df = pd.read_csv('Data/facebook_ads_electric_vehicles_with_openai_summaries_cached.csv')
        print(f"📊 Loaded dataset: {len(df)} rows")
        
        # Filter rows with valid image URLs
        df_with_images = df[df['new_image_url'].notna() & (df['new_image_url'] != '')].copy()
        print(f"🖼️  Found {len(df_with_images)} rows with image URLs")
        
        return df_with_images
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def download_all_images(df, base_dir, max_images=None, delay=1.0):
    """Download all images from the dataset"""
    
    if max_images:
        df = df.head(max_images)
        print(f"🎯 Limiting to first {max_images} images")
    
    total_images = len(df)
    successful_downloads = 0
    failed_downloads = 0
    failed_log = []
    
    print(f"\n🚀 Starting download of {total_images} images...")
    print(f"⏱️  Delay between downloads: {delay} seconds")
    
    for idx, row in df.iterrows():
        ad_id = row['ad_archive_id']
        image_url = row['new_image_url']
        
        # Generate filename
        filename = generate_filename(ad_id, image_url)
        filepath = os.path.join(base_dir, 'originals', filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"⏭️  Skipping {idx+1}/{total_images}: Already exists - {filename}")
            successful_downloads += 1
            continue
        
        print(f"📥 Downloading {idx+1}/{total_images}: {filename}")
        
        # Download the image
        success, message = download_image(image_url, filepath)
        
        if success:
            successful_downloads += 1
            print(f"✅ Success: {message}")
        else:
            failed_downloads += 1
            failed_log.append({
                'ad_id': ad_id,
                'url': image_url,
                'error': message
            })
            print(f"❌ Failed: {message}")
        
        # Progress update
        if (idx + 1) % 10 == 0:
            print(f"\n📊 Progress: {idx+1}/{total_images} processed")
            print(f"✅ Successful: {successful_downloads}")
            print(f"❌ Failed: {failed_downloads}")
            print("-" * 50)
        
        # Delay to be respectful to servers
        time.sleep(delay)
    
    # Save failed downloads log
    if failed_log:
        failed_df = pd.DataFrame(failed_log)
        failed_path = os.path.join(base_dir, 'failed', 'failed_downloads.csv')
        failed_df.to_csv(failed_path, index=False)
        print(f"📝 Failed downloads log saved to: {failed_path}")
    
    return successful_downloads, failed_downloads

def create_download_summary(base_dir, successful, failed):
    """Create a summary of the download process"""
    
    summary = {
        'total_attempted': successful + failed,
        'successful_downloads': successful,
        'failed_downloads': failed,
        'success_rate': (successful / (successful + failed)) * 100 if (successful + failed) > 0 else 0,
        'originals_dir': os.path.join(base_dir, 'originals'),
        'thumbnails_dir': os.path.join(base_dir, 'thumbnails'),
        'failed_log': os.path.join(base_dir, 'failed', 'failed_downloads.csv')
    }
    
    # Count actual files
    originals_count = len([f for f in os.listdir(summary['originals_dir']) if f.endswith(('.jpg', '.png', '.gif', '.webp'))])
    thumbnails_count = len([f for f in os.listdir(summary['thumbnails_dir']) if f.endswith(('.jpg', '.png', '.gif', '.webp'))])
    
    print(f"\n" + "="*60)
    print(f"📊 DOWNLOAD SUMMARY")
    print(f"="*60)
    print(f"🎯 Total attempted: {summary['total_attempted']}")
    print(f"✅ Successful: {summary['successful_downloads']}")
    print(f"❌ Failed: {summary['failed_downloads']}")
    print(f"📈 Success rate: {summary['success_rate']:.1f}%")
    print(f"\n📁 Files created:")
    print(f"   🖼️  Original images: {originals_count}")
    print(f"   🔍 Thumbnails: {thumbnails_count}")
    print(f"\n📂 Directories:")
    print(f"   📁 Originals: {summary['originals_dir']}")
    print(f"   📁 Thumbnails: {summary['thumbnails_dir']}")
    if failed > 0:
        print(f"   📝 Failed log: {summary['failed_log']}")
    
    return summary

def main():
    print("🖼️  Image Downloader for EV Ads Dataset")
    print("="*50)
    
    # Create directory structure
    base_dir = create_image_directory()
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Ask user for preferences
    print(f"\n🎯 Download Options:")
    print(f"1. Download all {len(df)} images")
    print(f"2. Download first 50 images (test)")
    print(f"3. Download first 200 images")
    print(f"4. Custom number")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    max_images = None
    if choice == "2":
        max_images = 50
    elif choice == "3":
        max_images = 200
    elif choice == "4":
        try:
            max_images = int(input("Enter number of images to download: "))
        except ValueError:
            print("Invalid number, downloading all images")
    
    # Download images
    successful, failed = download_all_images(df, base_dir, max_images)
    
    # Create summary
    create_download_summary(base_dir, successful, failed)

if __name__ == "__main__":
    main()
