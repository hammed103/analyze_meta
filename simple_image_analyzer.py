#!/usr/bin/env python3
"""
Simple image analysis for Facebook car ad images.
Basic image analysis without heavy AI dependencies.
"""

import pandas as pd
import requests
import os
import sys
from PIL import Image
import io
import time
from typing import List, Dict, Tuple
import json
from collections import Counter


class SimpleImageAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_image(self, url: str, timeout: int = 10) -> Image.Image:
        """Download image from URL and return PIL Image."""
        try:
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"Not an image: {content_type}")
            
            # Load image
            image_data = response.content
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None
    
    def analyze_basic_properties(self, image: Image.Image) -> Dict[str, any]:
        """Analyze basic image properties."""
        width, height = image.size
        aspect_ratio = width / height
        
        # Determine orientation
        if aspect_ratio > 1.3:
            orientation = "landscape"
        elif aspect_ratio < 0.8:
            orientation = "portrait"
        else:
            orientation = "square"
        
        # Determine size category
        total_pixels = width * height
        if total_pixels > 1000000:  # > 1MP
            size_category = "large"
        elif total_pixels > 300000:  # > 0.3MP
            size_category = "medium"
        else:
            size_category = "small"
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': round(aspect_ratio, 2),
            'orientation': orientation,
            'total_pixels': total_pixels,
            'size_category': size_category
        }
    
    def analyze_colors(self, image: Image.Image) -> Dict[str, any]:
        """Analyze color properties of the image."""
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get all pixels
        pixels = list(image.getdata())
        
        # Calculate average brightness
        brightness_values = [(r + g + b) / 3 for r, g, b in pixels]
        avg_brightness = sum(brightness_values) / len(brightness_values)
        
        # Determine brightness category
        if avg_brightness > 180:
            brightness_category = "bright"
        elif avg_brightness > 100:
            brightness_category = "medium"
        else:
            brightness_category = "dark"
        
        # Analyze color dominance (simplified)
        red_values = [r for r, g, b in pixels]
        green_values = [g for r, g, b in pixels]
        blue_values = [b for r, g, b in pixels]
        
        avg_red = sum(red_values) / len(red_values)
        avg_green = sum(green_values) / len(green_values)
        avg_blue = sum(blue_values) / len(blue_values)
        
        # Determine dominant color channel
        max_channel = max(avg_red, avg_green, avg_blue)
        if max_channel == avg_red:
            dominant_channel = "red"
        elif max_channel == avg_green:
            dominant_channel = "green"
        else:
            dominant_channel = "blue"
        
        # Calculate color variance (measure of colorfulness)
        color_variance = (
            sum([(r - avg_red)**2 for r in red_values]) +
            sum([(g - avg_green)**2 for g in green_values]) +
            sum([(b - avg_blue)**2 for b in blue_values])
        ) / (len(pixels) * 3)
        
        # Determine color richness
        if color_variance > 2000:
            color_richness = "high"
        elif color_variance > 800:
            color_richness = "medium"
        else:
            color_richness = "low"
        
        return {
            'avg_brightness': round(avg_brightness, 1),
            'brightness_category': brightness_category,
            'avg_red': round(avg_red, 1),
            'avg_green': round(avg_green, 1),
            'avg_blue': round(avg_blue, 1),
            'dominant_channel': dominant_channel,
            'color_variance': round(color_variance, 1),
            'color_richness': color_richness
        }
    
    def classify_basic_theme(self, image_properties: Dict, url: str) -> str:
        """Basic theme classification based on image properties and URL patterns."""
        # Simple heuristic-based classification
        width = image_properties.get('width', 0)
        height = image_properties.get('height', 0)
        aspect_ratio = image_properties.get('aspect_ratio', 1)
        brightness = image_properties.get('avg_brightness', 128)
        
        # URL-based hints
        url_lower = url.lower()
        
        # Check for specific patterns in URL
        if any(keyword in url_lower for keyword in ['interior', 'dashboard', 'cabin']):
            return "car_interior"
        elif any(keyword in url_lower for keyword in ['exterior', 'side', 'front', 'rear']):
            return "car_exterior"
        elif any(keyword in url_lower for keyword in ['charging', 'electric', 'ev']):
            return "electric_charging"
        elif any(keyword in url_lower for keyword in ['night', 'dark', 'lights']):
            return "night_scene"
        
        # Image property-based classification
        if aspect_ratio > 2.0:
            return "panoramic_view"
        elif aspect_ratio < 0.6:
            return "vertical_showcase"
        elif brightness < 80:
            return "dark_moody"
        elif brightness > 200:
            return "bright_clean"
        elif width > 1200 and height > 800:
            return "high_detail"
        else:
            return "standard_car_ad"
    
    def analyze_image(self, image_url: str, ad_metadata: Dict) -> Dict[str, any]:
        """Complete basic image analysis pipeline."""
        print(f"Analyzing: {ad_metadata.get('page_name', 'Unknown')} - {image_url[:50]}...")
        
        # Download image
        image = self.download_image(image_url)
        if image is None:
            return {
                'success': False,
                'error': 'Failed to download image',
                'url': image_url,
                **ad_metadata
            }
        
        # Perform analysis
        basic_props = self.analyze_basic_properties(image)
        color_props = self.analyze_colors(image)
        theme = self.classify_basic_theme({**basic_props, **color_props}, image_url)
        
        analysis = {
            'success': True,
            'url': image_url,
            'theme_classification': theme,
            **basic_props,
            **color_props,
            **ad_metadata
        }
        
        return analysis


def analyze_facebook_ad_images(csv_file: str, max_images: int = 100):
    """Analyze images from Facebook ads CSV."""
    print(f"Loading data from {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Get image URLs
    image_data = []
    for _, row in df.iterrows():
        # Check for image URLs in various columns
        image_url = None
        for col in ['first_image_url', 'image_0_original_url', 'snapshot_images']:
            if col in df.columns and pd.notna(row[col]) and row[col] != '':
                image_url = row[col]
                break
        
        if image_url:
            image_data.append({
                'url': image_url,
                'ad_id': row.get('ad_archive_id', ''),
                'page_name': row.get('page_name', ''),
                'car_model': row.get('matched_car_models', ''),
                'ad_title': row.get('ad_title', ''),
                'page_classification': row.get('page_classification', ''),
                'male_percentage': row.get('male_percentage', 0),
                'female_percentage': row.get('female_percentage', 0)
            })
    
    print(f"Found {len(image_data)} images to analyze")
    
    if len(image_data) == 0:
        print("No image URLs found in the data!")
        return
    
    # Limit number of images to analyze
    if len(image_data) > max_images:
        print(f"Limiting analysis to first {max_images} images")
        image_data = image_data[:max_images]
    
    # Initialize analyzer
    analyzer = SimpleImageAnalyzer()
    
    # Analyze images
    results = []
    for i, img_metadata in enumerate(image_data):
        print(f"\nAnalyzing image {i+1}/{len(image_data)}")
        
        analysis = analyzer.analyze_image(img_metadata['url'], img_metadata)
        results.append(analysis)
        
        # Add delay to be respectful
        time.sleep(0.5)
        
        # Save progress every 20 images
        if (i + 1) % 20 == 0:
            save_results(results, f"image_analysis_progress_{i+1}.json")
    
    # Save final results
    save_results(results, "facebook_ads_image_analysis_simple.json")
    
    # Create summary CSV
    create_summary_csv(results)
    
    # Print analysis summary
    print_analysis_summary(results)
    
    print(f"\nAnalysis complete! Analyzed {len(results)} images")
    print("Results saved to:")
    print("- facebook_ads_image_analysis_simple.json (detailed)")
    print("- facebook_ads_image_themes_simple.csv (summary)")


def save_results(results: List[Dict], filename: str):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Progress saved to {filename}")


def create_summary_csv(results: List[Dict]):
    """Create a summary CSV with key findings."""
    summary_data = []
    
    for result in results:
        if result['success']:
            summary_data.append({
                'ad_archive_id': result.get('ad_id', ''),
                'page_name': result.get('page_name', ''),
                'car_model': result.get('car_model', ''),
                'page_classification': result.get('page_classification', ''),
                'image_url': result['url'],
                'theme_classification': result.get('theme_classification', ''),
                'width': result.get('width', 0),
                'height': result.get('height', 0),
                'aspect_ratio': result.get('aspect_ratio', 0),
                'orientation': result.get('orientation', ''),
                'size_category': result.get('size_category', ''),
                'brightness_category': result.get('brightness_category', ''),
                'color_richness': result.get('color_richness', ''),
                'dominant_channel': result.get('dominant_channel', ''),
                'male_percentage': result.get('male_percentage', 0),
                'female_percentage': result.get('female_percentage', 0),
                'analysis_success': True
            })
        else:
            summary_data.append({
                'ad_archive_id': result.get('ad_id', ''),
                'page_name': result.get('page_name', ''),
                'car_model': result.get('car_model', ''),
                'image_url': result['url'],
                'error': result.get('error', ''),
                'analysis_success': False
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('facebook_ads_image_themes_simple.csv', index=False)


def print_analysis_summary(results: List[Dict]):
    """Print summary statistics of the analysis."""
    successful = [r for r in results if r.get('success', False)]
    
    if not successful:
        print("No successful analyses to summarize")
        return
    
    print(f"\n=== IMAGE ANALYSIS SUMMARY ===")
    print(f"Successfully analyzed: {len(successful)}/{len(results)} images")
    
    # Theme distribution
    themes = [r['theme_classification'] for r in successful]
    theme_counts = Counter(themes)
    print(f"\nTheme Distribution:")
    for theme, count in theme_counts.most_common():
        percentage = count / len(successful) * 100
        print(f"  {theme}: {count} images ({percentage:.1f}%)")
    
    # Orientation distribution
    orientations = [r['orientation'] for r in successful]
    orientation_counts = Counter(orientations)
    print(f"\nOrientation Distribution:")
    for orientation, count in orientation_counts.items():
        percentage = count / len(successful) * 100
        print(f"  {orientation}: {count} images ({percentage:.1f}%)")
    
    # Brightness distribution
    brightness_cats = [r['brightness_category'] for r in successful]
    brightness_counts = Counter(brightness_cats)
    print(f"\nBrightness Distribution:")
    for brightness, count in brightness_counts.items():
        percentage = count / len(successful) * 100
        print(f"  {brightness}: {count} images ({percentage:.1f}%)")
    
    # Color richness distribution
    color_richness = [r['color_richness'] for r in successful]
    richness_counts = Counter(color_richness)
    print(f"\nColor Richness Distribution:")
    for richness, count in richness_counts.items():
        percentage = count / len(successful) * 100
        print(f"  {richness}: {count} images ({percentage:.1f}%)")


def main():
    input_file = "facebook_ads_electric_vehicles_with_classifications.csv"
    max_images = 50  # Start with 50 images
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        max_images = int(sys.argv[2])
    
    print("=== SIMPLE FACEBOOK ADS IMAGE ANALYSIS ===")
    print(f"Input file: {input_file}")
    print(f"Max images to analyze: {max_images}")
    print("Note: This is a basic analysis. For AI-powered analysis, install the full dependencies.")
    
    analyze_facebook_ad_images(input_file, max_images)


if __name__ == "__main__":
    main()
