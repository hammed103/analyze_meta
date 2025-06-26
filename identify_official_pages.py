#!/usr/bin/env python3
"""
Identify official brand pages vs dealers/third-party advertisers in Facebook ads data.
Analyzes page names, URLs, and other indicators to classify pages.
"""

import pandas as pd
import re
import sys
from typing import Dict, List, Tuple


def get_brand_patterns() -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Define patterns for official brand pages vs dealers/third parties.
    """
    # Strict official brand patterns - only truly official pages
    official_patterns = {
        "tesla": [
            r"^tesla$",
            r"^tesla\s+(official|motors|inc)$",
            r"^tesla\s+(uk|usa|deutschland|france|italia)$",
        ],
        "volkswagen": [
            r"^volkswagen$",
            r"^vw$",
            r"^volkswagen\s+(official|group|ag)$",
            r"^vw\s+(official|group)$",
        ],
        "bmw": [
            r"^bmw$",
            r"^bmw\s+(official|group|ag)$",
            r"^bmw\s+(uk|usa|deutschland|france|italia)$",
        ],
        "audi": [
            r"^audi$",
            r"^audi\s+(official|ag)$",
            r"^audi\s+(uk|usa|deutschland|france|italia)$",
        ],
        "hyundai": [
            r"^hyundai$",
            r"^hyundai\s+(official|motor|motors|motor\s+group)$",
            r"^hyundai\s+(uk|usa|deutschland|france|italia)$",
        ],
        "volvo": [
            r"^volvo$",
            r"^volvo\s+(official|cars|car|group)$",
            r"^volvo\s+(uk|usa|deutschland|france|italia)$",
        ],
        "kia": [
            r"^kia$",
            r"^kia\s+(official|motors)$",
            r"^kia\s+(uk|usa|deutschland|france|italia)$",
        ],
        "renault": [
            r"^renault$",
            r"^renault\s+(official|group)$",
            r"^renault\s+(uk|usa|deutschland|france|italia)$",
        ],
        "mini": [
            r"^mini$",
            r"^mini\s+(official|cooper)$",
            r"^mini\s+(uk|usa|deutschland|france|italia)$",
        ],
        "cupra": [r"^cupra$", r"^cupra\s+(official)$"],
        "seat": [r"^seat$", r"^seat\s+(official)$"],
        "mg": [r"^mg$", r"^mg\s+(official|motor)$"],
    }

    # Dealer/distributor patterns - clearly dealerships
    dealer_patterns = [
        # Geographic dealers
        r"\b(bmw|audi|tesla|volkswagen|volvo|hyundai|kia|renault|mini|cupra|seat|mg)\s+(of|in)\s+[A-Z][a-z]+",
        r"\b[A-Z][a-z]+\s+(bmw|audi|tesla|volkswagen|volvo|hyundai|kia|renault|mini|cupra|seat|mg)\b",
        # Dealer names with brand
        r"\b(galpin|dimas|morrey|riverside|marshall|libertyville|hanlees|camelback|capistrano|garden\s+grove)\s+(bmw|audi|tesla|volkswagen|volvo|hyundai|kia|renault|mini|cupra|seat|mg)\b",
        # Dealer keywords
        r"\b(dealer|dealership|motors|automotive|auto|cars|sales|group|center|centre|zentrum)\b",
        # Service/parts companies
        r"\b(accessories|parts|mods|aftermarket|service|repair|maintenance)\b",
        # Rental/leasing
        r"\b(rental|rent|lease|leasing|hire|booking)\b",
        # Used car dealers
        r"\b(used|pre-owned|certified|approved|second\s+hand)\b",
    ]

    # Third-party patterns
    third_party_patterns = [
        r"\b(accessories|parts|mats|covers|charger|charging|mods|tuning)\b",
        r"\b(rental|rent|booking|hire)\b",
        r"\b(review|blog|news|media|magazine)\b",
        r"\b(tuning|modification|custom|aftermarket)\b",
        r"\b(deals|discount|best\s+deals|top\s+car)\b",
        r"\b(прокат|租车|location|alquiler)\b",  # rental in other languages
    ]

    return official_patterns, dealer_patterns, third_party_patterns


def classify_page(
    page_name: str, page_profile_uri: str = None, advertiser_name: str = None
) -> Tuple[str, str, float]:
    """
    Classify a page as official, dealer, or third-party.
    Returns (classification, reason, confidence_score)
    """
    if pd.isna(page_name) or page_name == "":
        return "unknown", "No page name", 0.0

    page_lower = page_name.lower().strip()
    official_patterns, dealer_patterns, third_party_patterns = get_brand_patterns()

    # First check for dealer patterns (most common and specific)
    for pattern in dealer_patterns:
        if re.search(pattern, page_lower, re.IGNORECASE):
            confidence = 0.9
            return "dealer", f"Matches dealer pattern: {pattern}", confidence

    # Then check for third-party patterns
    for pattern in third_party_patterns:
        if re.search(pattern, page_lower, re.IGNORECASE):
            return "third_party", f"Matches third-party pattern: {pattern}", 0.8

    # Finally check for official brand patterns (most strict)
    for brand, patterns in official_patterns.items():
        for pattern in patterns:
            if re.search(pattern, page_lower, re.IGNORECASE):
                # Additional checks for higher confidence
                confidence = 0.8

                # Boost confidence if URL suggests official
                if page_profile_uri and any(
                    domain in str(page_profile_uri).lower()
                    for domain in [
                        f"{brand}.com",
                        f"{brand}.co.uk",
                        f"{brand}.de",
                        f"{brand}.fr",
                    ]
                ):
                    confidence += 0.15

                # Boost confidence if advertiser name matches
                if advertiser_name and brand in str(advertiser_name).lower():
                    confidence += 0.05

                return (
                    "official",
                    f"Matches {brand} official pattern: {pattern}",
                    min(confidence, 1.0),
                )

    # Default classification based on complexity
    if len(page_name.split()) == 1 and any(
        brand in page_lower for brand in official_patterns.keys()
    ):
        return "likely_official", "Simple brand name", 0.5

    return "unknown", "No clear indicators", 0.3


def analyze_official_pages(csv_file: str):
    """
    Analyze the CSV to identify official pages vs dealers/third parties.
    """
    print(f"Loading data from {csv_file}...")

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("\nClassifying pages...")

    # Apply classification
    classifications = []
    reasons = []
    confidence_scores = []

    for _, row in df.iterrows():
        page_name = row.get("page_name", "")
        page_uri = row.get("page_profile_uri", "")
        advertiser = row.get("advertiser_name", "")

        classification, reason, confidence = classify_page(
            page_name, page_uri, advertiser
        )
        classifications.append(classification)
        reasons.append(reason)
        confidence_scores.append(confidence)

    # Add classification columns
    df["page_classification"] = classifications
    df["classification_reason"] = reasons
    df["classification_confidence"] = confidence_scores

    # Analysis results
    print(f"\n=== PAGE CLASSIFICATION RESULTS ===")
    classification_counts = df["page_classification"].value_counts()
    for classification, count in classification_counts.items():
        percentage = count / len(df) * 100
        print(f"{classification}: {count} pages ({percentage:.1f}%)")

    # Show official pages
    official_pages = df[df["page_classification"] == "official"]
    if len(official_pages) > 0:
        print(f"\n=== IDENTIFIED OFFICIAL BRAND PAGES ===")
        official_summary = (
            official_pages.groupby("page_name")
            .agg(
                {
                    "ad_archive_id": "count",
                    "matched_car_models": lambda x: "; ".join(x.unique()),
                    "classification_confidence": "mean",
                }
            )
            .round(2)
        )
        official_summary.columns = ["ad_count", "models_advertised", "avg_confidence"]
        official_summary = official_summary.sort_values("ad_count", ascending=False)

        for page, data in official_summary.head(10).iterrows():
            print(
                f"{page}: {data['ad_count']} ads, Models: {data['models_advertised'][:50]}..."
            )

    # Show likely official pages
    likely_official = df[df["page_classification"] == "likely_official"]
    if len(likely_official) > 0:
        print(f"\n=== LIKELY OFFICIAL PAGES (need verification) ===")
        likely_summary = (
            likely_official.groupby("page_name")["ad_archive_id"]
            .count()
            .sort_values(ascending=False)
        )
        for page, count in likely_summary.head(10).items():
            print(f"{page}: {count} ads")

    # Show top dealers
    dealers = df[df["page_classification"] == "dealer"]
    if len(dealers) > 0:
        print(f"\n=== TOP DEALERS/DISTRIBUTORS ===")
        dealer_summary = (
            dealers.groupby("page_name")["ad_archive_id"]
            .count()
            .sort_values(ascending=False)
        )
        for page, count in dealer_summary.head(10).items():
            print(f"{page}: {count} ads")

    # Show third parties
    third_parties = df[df["page_classification"] == "third_party"]
    if len(third_parties) > 0:
        print(f"\n=== THIRD-PARTY ADVERTISERS ===")
        third_party_summary = (
            third_parties.groupby("page_name")["ad_archive_id"]
            .count()
            .sort_values(ascending=False)
        )
        for page, count in third_party_summary.head(10).items():
            print(f"{page}: {count} ads")

    # Save enhanced data
    output_file = csv_file.replace(".csv", "_with_classifications.csv")
    df.to_csv(output_file, index=False)
    print(f"\nEnhanced data saved to: {output_file}")

    # Summary by brand and page type
    print(f"\n=== BRAND ANALYSIS ===")
    for model in df["matched_car_models"].unique():
        if pd.notna(model) and model != "":
            model_data = df[df["matched_car_models"].str.contains(model, na=False)]
            if len(model_data) > 0:
                official_count = len(
                    model_data[model_data["page_classification"] == "official"]
                )
                dealer_count = len(
                    model_data[model_data["page_classification"] == "dealer"]
                )
                third_party_count = len(
                    model_data[model_data["page_classification"] == "third_party"]
                )

                print(f"{model}:")
                print(f"  Official: {official_count} ads")
                print(f"  Dealers: {dealer_count} ads")
                print(f"  Third-party: {third_party_count} ads")


def main():
    input_file = "facebook_ads_electric_vehicles.csv"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    print("=== OFFICIAL PAGE IDENTIFICATION ===")
    print(f"Analyzing: {input_file}")

    analyze_official_pages(input_file)


if __name__ == "__main__":
    main()
