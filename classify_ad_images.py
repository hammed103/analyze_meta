#!/usr/bin/env python3
"""
AI-powered image classification for Facebook car ad images.
Uses computer vision models to analyze themes, objects, and content.
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

# For AI image analysis - we'll use multiple approaches
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch pillow")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: opencv not available. Install with: pip install opencv-python")


class ImageAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Initialize AI models if available
        self.blip_processor = None
        self.blip_model = None
        self.clip_processor = None
        self.clip_model = None
        
        if TRANSFORMERS_AVAILABLE:
            self.load_models()
    
    def load_models(self):
        """Load AI models for image analysis."""
        try:
            print("Loading BLIP model for image captioning...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            print("Loading CLIP model for image classification...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.blip_processor = None
            self.blip_model = None
            self.clip_processor = None
            self.clip_model = None
    
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
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate descriptive caption for the image using BLIP."""
        if not self.blip_processor or not self.blip_model:
            return "Caption generation not available"
        
        try:
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"Error generating caption: {e}"
    
    def classify_theme(self, image: Image.Image) -> Dict[str, float]:
        """Classify image theme using CLIP with predefined categories."""
        if not self.clip_processor or not self.clip_model:
            return {"classification_not_available": 1.0}
        
        # Define car ad themes to classify
        themes = [
            "car in urban city environment",
            "car in nature landscape",
            "car interior dashboard",
            "car exterior side view",
            "car front view",
            "car charging electric vehicle",
            "family with car",
            "person driving car",
            "car in garage or showroom",
            "car on highway or road",
            "luxury car lifestyle",
            "car technology features",
            "car safety features",
            "car performance racing",
            "car in parking lot",
            "car at night with lights"
        ]
        
        try:
            inputs = self.clip_processor(text=themes, images=image, return_tensors="pt", padding=True)
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Create theme scores dictionary
            theme_scores = {}
            for i, theme in enumerate(themes):
                theme_scores[theme] = float(probs[0][i])
            
            return theme_scores
            
        except Exception as e:
            return {"error": f"Classification error: {e}"}
    
    def analyze_colors(self, image: Image.Image) -> Dict[str, any]:
        """Analyze dominant colors in the image."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Get dominant colors using k-means clustering
            if OPENCV_AVAILABLE:
                # Reshape image to be a list of pixels
                pixels = img_array.reshape((-1, 3))
                pixels = np.float32(pixels)
                
                # Apply k-means to find dominant colors
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                k = 5  # Number of dominant colors
                _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                # Convert back to uint8 and get color percentages
                centers = np.uint8(centers)
                
                # Count pixels for each cluster
                unique_labels, counts = np.unique(labels, return_counts=True)
                percentages = counts / len(pixels) * 100
                
                dominant_colors = []
                for i, (color, percentage) in enumerate(zip(centers, percentages)):
                    dominant_colors.append({
                        'color_rgb': color.tolist(),
                        'percentage': float(percentage)
                    })
                
                # Sort by percentage
                dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
                
                return {
                    'dominant_colors': dominant_colors[:3],  # Top 3 colors
                    'brightness': float(np.mean(img_array)),
                    'contrast': float(np.std(img_array))
                }
            else:
                # Simple brightness analysis without opencv
                return {
                    'brightness': float(np.mean(img_array)),
                    'contrast': float(np.std(img_array)),
                    'dominant_colors': []
                }
                
        except Exception as e:
            return {'error': f"Color analysis error: {e}"}
    
    def analyze_image(self, image_url: str) -> Dict[str, any]:
        """Complete image analysis pipeline."""
        print(f"Analyzing image: {image_url[:50]}...")
        
        # Download image
        image = self.download_image(image_url)
        if image is None:
            return {
                'success': False,
                'error': 'Failed to download image',
                'url': image_url
            }
        
        # Perform analysis
        analysis = {
            'success': True,
            'url': image_url,
            'image_size': image.size,
            'caption': self.generate_caption(image),
            'theme_classification': self.classify_theme(image),
            'color_analysis': self.analyze_colors(image)
        }
        
        # Get top theme
        if 'error' not in analysis['theme_classification']:
            top_theme = max(analysis['theme_classification'].items(), key=lambda x: x[1])
            analysis['top_theme'] = top_theme[0]
            analysis['top_theme_confidence'] = top_theme[1]
        else:
            analysis['top_theme'] = 'unknown'
            analysis['top_theme_confidence'] = 0.0
        
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
    image_urls = []
    for _, row in df.iterrows():
        # Check for image URLs in various columns
        for col in ['first_image_url', 'image_0_original_url', 'snapshot_images']:
            if col in df.columns and pd.notna(row[col]) and row[col] != '':
                image_urls.append({
                    'url': row[col],
                    'ad_id': row.get('ad_archive_id', ''),
                    'page_name': row.get('page_name', ''),
                    'car_model': row.get('matched_car_models', ''),
                    'ad_title': row.get('ad_title', '')
                })
                break
    
    print(f"Found {len(image_urls)} images to analyze")
    
    if len(image_urls) == 0:
        print("No image URLs found in the data!")
        return
    
    # Limit number of images to analyze
    if len(image_urls) > max_images:
        print(f"Limiting analysis to first {max_images} images")
        image_urls = image_urls[:max_images]
    
    # Initialize analyzer
    analyzer = ImageAnalyzer()
    
    # Analyze images
    results = []
    for i, img_data in enumerate(image_urls):
        print(f"\nAnalyzing image {i+1}/{len(image_urls)}")
        print(f"Ad: {img_data['page_name']} - {img_data['car_model']}")
        
        analysis = analyzer.analyze_image(img_data['url'])
        analysis.update(img_data)  # Add metadata
        results.append(analysis)
        
        # Add delay to be respectful
        time.sleep(1)
        
        # Save progress every 10 images
        if (i + 1) % 10 == 0:
            save_results(results, f"image_analysis_progress_{i+1}.json")
    
    # Save final results
    save_results(results, "facebook_ads_image_analysis.json")
    
    # Create summary CSV
    create_summary_csv(results)
    
    print(f"\nAnalysis complete! Analyzed {len(results)} images")
    print("Results saved to:")
    print("- facebook_ads_image_analysis.json (detailed)")
    print("- facebook_ads_image_themes.csv (summary)")


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
                'ad_title': result.get('ad_title', ''),
                'image_url': result['url'],
                'image_caption': result.get('caption', ''),
                'top_theme': result.get('top_theme', ''),
                'theme_confidence': result.get('top_theme_confidence', 0),
                'image_width': result.get('image_size', [0, 0])[0],
                'image_height': result.get('image_size', [0, 0])[1],
                'brightness': result.get('color_analysis', {}).get('brightness', 0),
                'contrast': result.get('color_analysis', {}).get('contrast', 0),
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
    summary_df.to_csv('facebook_ads_image_themes.csv', index=False)
    
    # Print summary statistics
    if len(summary_data) > 0:
        successful = len([r for r in summary_data if r.get('analysis_success', False)])
        print(f"\nSUMMARY STATISTICS:")
        print(f"Successfully analyzed: {successful}/{len(summary_data)} images")
        
        if successful > 0:
            themes = [r['top_theme'] for r in summary_data if r.get('top_theme')]
            if themes:
                theme_counts = pd.Series(themes).value_counts()
                print(f"\nTop image themes:")
                for theme, count in theme_counts.head(10).items():
                    print(f"  {theme}: {count} images")


def main():
    input_file = "facebook_ads_electric_vehicles_with_classifications.csv"
    max_images = 50  # Start with 50 images for testing
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        max_images = int(sys.argv[2])
    
    print("=== FACEBOOK ADS IMAGE ANALYSIS ===")
    print(f"Input file: {input_file}")
    print(f"Max images to analyze: {max_images}")
    
    if not TRANSFORMERS_AVAILABLE:
        print("\nWARNING: AI models not available!")
        print("Install required packages:")
        print("pip install transformers torch pillow")
        print("pip install opencv-python  # optional for color analysis")
        return
    
    analyze_facebook_ad_images(input_file, max_images)


if __name__ == "__main__":
    main()
