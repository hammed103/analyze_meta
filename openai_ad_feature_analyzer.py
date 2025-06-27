#!/usr/bin/env python3
"""
OpenAI Ad Feature Analyzer
Sends all text fields to OpenAI to analyze car ads for specific features and provide summaries
"""

import pandas as pd
import requests
import json
import time
import os
import sys
from typing import Dict, List, Optional


class OpenAIAdAnalyzer:
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key."""
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def create_analysis_prompt(self, ad_data: Dict) -> str:
        """Create a comprehensive prompt for analyzing the ad."""

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
        if ad_data.get("ad_theme"):
            text_fields.append(f"Visual Theme: {ad_data['ad_theme']}")

        combined_text = "\n".join(text_fields)

        prompt = f"""
Analyze this car advertisement and provide a comprehensive summary focusing on the key features mentioned.

CAR MODEL: {ad_data.get('matched_car_models', 'Unknown')}
ADVERTISER TYPE: {ad_data.get('page_classification', 'Unknown')}

ADVERTISEMENT CONTENT:
{combined_text}

Please analyze this ad and provide a summary in the following format:

**Brand & Product Focus:** [What brand/model is being advertised and who it's targeting]

**Key Message/Slogan:** [Main headline, slogan, or key message if present]

**Dealership/Advertiser:** [Information about the advertiser - official brand, dealer, etc.]

Then, ONLY include the following sections IF they are mentioned in the ad:

**Range and Charging:** [Details about battery range, charging speed, charging infrastructure, etc.]

**Performance:** [Details about acceleration, power, handling, driving modes, etc.]

**Interior and Comfort:** [Details about space, seating, climate control, luxury features, etc.]

**Infotainment & Audio:** [Details about screens, connectivity, entertainment, navigation, etc.]

**Exterior Design:** [Details about styling, lights, wheels, aerodynamics, visual appeal, etc.]

**Safety & Assistance:** [Details about safety ratings, driver aids, autonomous features, etc.]

**Connectivity and Digital Experience:** [Details about apps, wireless features, smart integration, etc.]

**Overall Theme:** [The overall style, tone, and approach of the advertisement]

Important: Only include feature sections that are actually mentioned or emphasized in the ad content. If a category isn't mentioned, don't include that section at all.
"""
        return prompt

    def analyze_ad(self, ad_data: Dict) -> Dict:
        """Analyze a single ad using OpenAI."""

        prompt = self.create_analysis_prompt(ad_data)

        payload = {
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3000,
            "temperature": 0.1,
        }

        try:
            response = requests.post(
                self.base_url, headers=self.headers, json=payload, timeout=30
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Simple text response - no JSON parsing needed
            return {
                "success": True,
                "summary": content.strip(),
                "raw_response": content,
                **ad_data,
            }

        except Exception as e:
            return {"success": False, "error": str(e), **ad_data}


def analyze_ads_dataset(csv_file: str, api_key: str, max_ads: int = 50):
    """Analyze ads dataset using OpenAI."""

    print(f"🚗⚡ OPENAI AD FEATURE ANALYSIS")
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

    # Initialize analyzer
    analyzer = OpenAIAdAnalyzer(api_key)

    # Process ads
    results = []
    total_cost_estimate = 0

    for i, (_, row) in enumerate(text_ads.iterrows()):
        print(f"\n🔍 Analyzing ad {i+1}/{len(text_ads)}")
        print(f"   Advertiser: {row.get('page_name', 'Unknown')}")
        print(f"   Model: {row.get('matched_car_models', 'Unknown')}")

        # Prepare ad data
        ad_data = {
            "ad_archive_id": row.get("ad_archive_id", ""),
            "page_name": row.get("page_name", ""),
            "matched_car_models": row.get("matched_car_models", ""),
            "page_classification": row.get("page_classification", ""),
            "ad_title": row.get("ad_title", ""),
            "ad_text": row.get("ad_text", ""),
            "cta_text": row.get("cta_text", ""),
            "extracted_text": row.get("extracted_text", ""),
            "ad_theme": row.get("ad_theme", ""),
        }

        # Analyze ad
        result = analyzer.analyze_ad(ad_data)
        results.append(result)

        # Estimate cost (rough: $0.005 per analysis)
        if result["success"]:
            total_cost_estimate += 0.005
            print(f"   ✅ Analysis complete")
        else:
            print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")

        # Rate limiting
        if i < len(text_ads) - 1:
            time.sleep(1)

        # Save progress every 10 ads
        if (i + 1) % 10 == 0:
            save_results(results, f"openai_analysis_progress_{i+1}.json")

    # Save final results
    save_results(results, "openai_ad_analysis_complete.json")
    create_analysis_csv(results)

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📊 Processed: {len(results)} ads")
    print(f"✅ Successful: {len([r for r in results if r['success']])}")
    print(f"❌ Failed: {len([r for r in results if not r['success']])}")
    print(f"💰 Estimated cost: ${total_cost_estimate:.2f}")
    print(f"\n📁 Results saved to:")
    print(f"   - openai_ad_analysis_complete.json (detailed)")
    print(f"   - openai_ad_summary.csv (summaries)")


def save_results(results: List[Dict], filename: str):
    """Save results to JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Progress saved to {filename}")


def create_analysis_csv(results: List[Dict]):
    """Create a simple CSV from OpenAI analysis results."""
    csv_data = []

    for result in results:
        if result["success"] and result.get("summary"):
            # Simple structure with just the summary
            row = {
                "ad_archive_id": result.get("ad_archive_id", ""),
                "page_name": result.get("page_name", ""),
                "matched_car_models": result.get("matched_car_models", ""),
                "page_classification": result.get("page_classification", ""),
                "openai_summary": result["summary"],
            }
            csv_data.append(row)
        else:
            # Add failed analysis with basic info
            csv_data.append(
                {
                    "ad_archive_id": result.get("ad_archive_id", ""),
                    "page_name": result.get("page_name", ""),
                    "matched_car_models": result.get("matched_car_models", ""),
                    "analysis_error": result.get("error", "Analysis failed"),
                }
            )

    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv("openai_ad_summary.csv", index=False, encoding="utf-8")
    print("📊 OpenAI summaries saved to openai_ad_summary.csv")


def main():
    """Main function."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr pass it as an argument:")
        print("python3 openai_ad_feature_analyzer.py your-api-key-here")

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

    # Run analysis
    analyze_ads_dataset(csv_file, api_key, max_ads)


if __name__ == "__main__":
    main()
