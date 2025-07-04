#!/usr/bin/env python3
"""
Car Feature Extractor
Analyzes car advertisements to extract specific car features mentioned.
Returns null when no car features are mentioned.
"""

import pandas as pd
import requests
import json
import time
import os
import sys
from typing import Dict, List, Optional


class CarFeatureExtractor:
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key."""
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def create_feature_extraction_prompt(self, ad_data: Dict) -> str:
        """Create a prompt specifically for car feature extraction."""

        # Collect all text fields
        text_fields = []
        if ad_data.get("ad_title"):
            text_fields.append(f"Title: {ad_data['ad_title']}")
        if ad_data.get("ad_text"):
            text_fields.append(f"Ad Text: {ad_data['ad_text']}")
        if ad_data.get("cta_text"):
            text_fields.append(f"Call-to-Action: {ad_data['cta_text']}")
        if ad_data.get("page_name"):
            text_fields.append(f"Advertiser: {ad_data['page_name']}")
        if ad_data.get("extracted_text"):
            text_fields.append(
                f"Extracted Text from Image: {ad_data['extracted_text']}"
            )

        combined_text = "\n".join(text_fields)

        prompt = f"""
Analyze this car advertisement and extract ONLY the specific car features that are mentioned.

CAR MODEL: {ad_data.get('matched_car_models', 'Unknown')}
ADVERTISER TYPE: {ad_data.get('page_classification', 'Unknown')}

ADVERTISEMENT CONTENT:
{combined_text}

TASK: Extract any car features mentioned in the ad content.

INSTRUCTIONS:
- If NO car features are mentioned, respond with exactly: "null"
- If car features ARE mentioned, list them clearly
- Focus on technical specifications and tangible features, not marketing language
- Be specific about the features mentioned, don't infer or assume

Remember: If no actual car features are mentioned, respond with "null"
"""
        return prompt

    def extract_features(self, ad_data: Dict) -> Dict:
        """Extract car features from a single ad using OpenAI."""

        prompt = self.create_feature_extraction_prompt(ad_data)

        payload = {
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.1,
        }

        try:
            response = requests.post(
                self.base_url, headers=self.headers, json=payload, timeout=30
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # Check if no features were found
            if content.lower() == "null" or content.lower() == '"null"':
                features_found = None
            else:
                features_found = content

            return {
                "success": True,
                "car_features": features_found,
                "raw_response": content,
                **ad_data,
            }

        except Exception as e:
            return {"success": False, "error": str(e), **ad_data}


def extract_features_from_dataset(csv_file: str, api_key: str, max_ads: int = 50):
    """Extract car features from ads dataset using OpenAI."""

    print(f"🚗🔍 CAR FEATURE EXTRACTION")
    print(f"=" * 50)

    # Load data
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} ad records")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Filter for ads with sufficient text content
    text_columns = ["ad_title", "ad_text", "cta_text", "extracted_text"]
    df["has_text_content"] = df[text_columns].notna().any(axis=1)

    text_ads = df[df["has_text_content"]].copy()
    print(f"📝 Found {len(text_ads)} ads with text content")

    if len(text_ads) == 0:
        print("❌ No ads with text content found!")
        return

    # Limit analysis
    if len(text_ads) > max_ads:
        print(f"🔢 Limiting analysis to {max_ads} ads")
        text_ads = text_ads.head(max_ads)

    # Initialize extractor
    extractor = CarFeatureExtractor(api_key)

    # Process ads
    results = []
    total_cost_estimate = 0
    features_found_count = 0

    for i, (_, row) in enumerate(text_ads.iterrows()):
        print(f"\n🔍 Analyzing ad {i+1}/{len(text_ads)}")
        print(f"   Advertiser: {row.get('page_name', 'Unknown')}")
        print(f"   Model: {row.get('matched_car_models', 'Unknown')}")

        # Prepare ad data
        ad_data = {
            "ad_title": row.get("ad_title", ""),
            "ad_text": row.get("ad_text", ""),
            "cta_text": row.get("cta_text", ""),
            "extracted_text": row.get("extracted_text", ""),
            # "ad_theme": row.get("ad_theme", ""),
        }

        # Extract features
        result = extractor.extract_features(ad_data)
        results.append(result)

        # Track results
        if result["success"]:
            total_cost_estimate += 0.003  # Lower cost estimate for feature extraction
            if result["car_features"] is not None:
                features_found_count += 1
                print(f"   ✅ Car features found")
            else:
                print(f"   ⚪ No car features mentioned")
        else:
            print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")

        # Rate limiting
        if i < len(text_ads) - 1:
            time.sleep(1)

        # Save progress every 10 ads
        if (i + 1) % 10 == 0:
            save_feature_results(results, f"car_features_progress_{i+1}.json")

    # Save final results
    save_feature_results(results, "car_features_complete.json")
    create_features_csv(results)

    print(f"\n🎉 FEATURE EXTRACTION COMPLETE!")
    print(f"📊 Processed: {len(results)} ads")
    print(f"✅ Successful: {len([r for r in results if r['success']])}")
    print(f"🚗 Features found: {features_found_count}")
    print(
        f"⚪ No features: {len([r for r in results if r['success'] and r['car_features'] is None])}"
    )
    print(f"❌ Failed: {len([r for r in results if not r['success']])}")
    print(f"💰 Estimated cost: ${total_cost_estimate:.2f}")
    print(f"\n📁 Results saved to:")
    print(f"   - car_features_complete.json (detailed)")
    print(f"   - car_features_extracted.csv (features only)")


def save_feature_results(results: List[Dict], filename: str):
    """Save results to JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Progress saved to {filename}")


def create_features_csv(results: List[Dict]):
    """Create a CSV from car feature extraction results."""
    csv_data = []

    for result in results:
        if result["success"]:
            row = {
                "ad_archive_id": result.get("ad_archive_id", ""),
                "page_name": result.get("page_name", ""),
                "matched_car_models": result.get("matched_car_models", ""),
                "page_classification": result.get("page_classification", ""),
                "car_features": result.get(
                    "car_features"
                ),  # Will be None if no features
                "features_found": (
                    "Yes" if result.get("car_features") is not None else "No"
                ),
            }
            csv_data.append(row)
        else:
            # Add failed analysis with basic info
            csv_data.append(
                {
                    "ad_archive_id": result.get("ad_archive_id", ""),
                    "page_name": result.get("page_name", ""),
                    "matched_car_models": result.get("matched_car_models", ""),
                    "car_features": None,
                    "features_found": "Error",
                    "extraction_error": result.get("error", "Analysis failed"),
                }
            )

    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv("car_features_extracted.csv", index=False, encoding="utf-8")
    print("📊 Car features saved to car_features_extracted.csv")


def main():
    """Main function."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr pass it as an argument:")
        print("python3 car_feature_extractor.py your-api-key-here")

        if len(sys.argv) > 1:
            api_key = sys.argv[1]
        else:
            return

    # Input file
    csv_file = "facebook_ads_electric_vehicles_enhanced_with_gpt4.csv"
    max_ads = 25  # Start with 25 ads for testing

    if len(sys.argv) > 2:
        max_ads = int(sys.argv[2])

    print(f"🔑 API Key: {'*' * (len(api_key) - 8) + api_key[-8:]}")
    print(f"📁 Input file: {csv_file}")
    print(f"🔢 Max ads: {max_ads}")

    # Run feature extraction
    extract_features_from_dataset(csv_file, api_key, max_ads)


if __name__ == "__main__":
    main()
