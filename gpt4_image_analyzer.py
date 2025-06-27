#!/usr/bin/env python3
"""
GPT-4 Mini Image Analyzer for Facebook Car Ads
Uses OpenAI GPT-4 Vision to analyze ad images and extract text and themes
"""

import pandas as pd
import requests
import base64
import json
import time
import os
from typing import Dict, List, Optional
import sys
from PIL import Image
import io


class GPT4ImageAnalyzer:
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key."""
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # Session for downloading images
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def encode_image_from_url(
        self, image_url: str, max_size: tuple = (1024, 1024)
    ) -> Optional[str]:
        """Download and encode image from URL to base64."""
        try:
            response = self.session.get(image_url, timeout=10)
            response.raise_for_status()

            # Open and resize image if needed
            image = Image.open(io.BytesIO(response.content))

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too large (to save API costs)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            image_data = buffer.getvalue()

            return base64.b64encode(image_data).decode("utf-8")

        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
            return None

    def analyze_ad_image(self, image_url: str, ad_metadata: Dict) -> Dict:
        """Analyze a single ad image using GPT-4 Vision."""

        # Encode image
        base64_image = self.encode_image_from_url(image_url)
        if not base64_image:
            return {
                "success": False,
                "error": "Failed to process image",
                "image_url": image_url,
                **ad_metadata,
            }

        # Create the prompt - much simpler!
        prompt = f"""
Look at this car advertisement image and extract:

1. ALL TEXT visible in the image (brand names, model names, headlines, slogans, prices, contact info, specifications, etc.)

2. Brief description of the overall theme/style of the ad

Please respond in this simple format:

TEXT FOUND:
[List all text you can see in the image]

THEME:
[Brief description of the ad's theme, style, and visual approach - 2-3 sentences max]

Context: This is a Facebook ad for "{ad_metadata.get('car_model', 'Unknown')}" by "{ad_metadata.get('page_name', 'Unknown')}".
"""

        # Prepare the API request
        payload = {
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.1,
        }

        try:
            # Make API request
            response = requests.post(
                self.base_url, headers=self.headers, json=payload, timeout=30
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Simple text response - no JSON parsing needed
            return {
                "success": True,
                "image_url": image_url,
                "text_analysis": content,
                **ad_metadata,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_url": image_url,
                **ad_metadata,
            }


def analyze_facebook_ads_with_gpt4(csv_file: str, api_key: str, max_images: int = 50):
    """Analyze Facebook ads images using GPT-4 Vision."""

    print(f"🚗⚡ GPT-4 MINI IMAGE ANALYSIS")
    print(f"=" * 50)

    # Load data
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} ad records")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Filter for ads with images
    image_ads = df[df["first_image_url"].notna()].copy()
    print(f"📸 Found {len(image_ads)} ads with images")

    if len(image_ads) == 0:
        print("❌ No ads with images found!")
        return

    # Limit analysis
    if len(image_ads) > max_images:
        print(f"🔢 Limiting analysis to {max_images} images")
        image_ads = image_ads.head(max_images)

    # Initialize analyzer
    analyzer = GPT4ImageAnalyzer(api_key)

    # Process images
    results = []
    total_cost_estimate = 0

    for i, (_, row) in enumerate(image_ads.iterrows()):
        print(f"\n🔍 Analyzing image {i+1}/{len(image_ads)}")
        print(f"   Advertiser: {row['page_name']}")
        print(f"   Model: {row['matched_car_models']}")

        # Prepare metadata
        metadata = {
            "ad_id": row.get("ad_archive_id", ""),
            "page_name": row.get("page_name", ""),
            "car_model": row.get("matched_car_models", ""),
            "ad_title": row.get("ad_title", ""),
            "cta_text": row.get("cta_text", ""),
            "page_classification": row.get("page_classification", ""),
        }

        # Analyze image
        result = analyzer.analyze_ad_image(row["first_image_url"], metadata)
        results.append(result)

        # Estimate cost (rough: $0.01 per image for GPT-4 Vision)
        if result["success"]:
            total_cost_estimate += 0.01
            print(f"   ✅ Analysis complete")
        else:
            print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")

        # Rate limiting - be respectful to OpenAI API
        if i < len(image_ads) - 1:
            time.sleep(1)

        # Save progress every 10 images
        if (i + 1) % 10 == 0:
            save_results(results, f"gpt4_analysis_progress_{i+1}.json")

    # Save final results
    save_results(results, "gpt4_image_analysis_complete.json")
    create_analysis_csv(results)

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📊 Processed: {len(results)} images")
    print(f"✅ Successful: {len([r for r in results if r['success']])}")
    print(f"❌ Failed: {len([r for r in results if not r['success']])}")
    print(f"💰 Estimated cost: ${total_cost_estimate:.2f}")
    print(f"\n📁 Results saved to:")
    print(f"   - gpt4_image_analysis_complete.json (detailed)")
    print(f"   - gpt4_analysis_summary.csv (structured)")


def save_results(results: List[Dict], filename: str):
    """Save results to JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Progress saved to {filename}")


def create_analysis_csv(results: List[Dict]):
    """Create a simple CSV from GPT-4 text analysis results."""
    csv_data = []

    for result in results:
        if result["success"] and result.get("text_analysis"):
            # Simple structure with just the text analysis
            row = {
                "ad_id": result.get("ad_id", ""),
                "page_name": result.get("page_name", ""),
                "car_model": result.get("car_model", ""),
                "page_classification": result.get("page_classification", ""),
                "image_url": result["image_url"],
                "gpt4_text_analysis": result["text_analysis"],
            }
            csv_data.append(row)
        else:
            # Add failed analysis with basic info
            csv_data.append(
                {
                    "ad_id": result.get("ad_id", ""),
                    "page_name": result.get("page_name", ""),
                    "car_model": result.get("car_model", ""),
                    "image_url": result["image_url"],
                    "analysis_error": result.get("error", "Analysis failed"),
                }
            )

    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv("gpt4_analysis_summary.csv", index=False, encoding="utf-8")
    print("📊 Simple text analysis saved to gpt4_analysis_summary.csv")


def main():
    """Main function."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr pass it as an argument:")
        print("python3 gpt4_image_analyzer.py your-api-key-here")

        if len(sys.argv) > 1:
            api_key = sys.argv[1]
        else:
            return

    # Input file
    csv_file = "facebook_ads_electric_vehicles.csv"
    max_images = 25  # Start with 25 images for testing

    if len(sys.argv) > 2:
        max_images = int(sys.argv[2])

    print(f"🔑 API Key: {'*' * (len(api_key) - 8) + api_key[-8:]}")
    print(f"📁 Input file: {csv_file}")
    print(f"🔢 Max images: {max_images}")

    # Run analysis
    analyze_facebook_ads_with_gpt4(csv_file, api_key, max_images)


if __name__ == "__main__":
    main()
