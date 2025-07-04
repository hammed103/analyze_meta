#!/usr/bin/env python3
"""
Car Features Analysis
Analyzes extracted car features by different car models and creates comprehensive reports
"""

import pandas as pd
import json
from collections import defaultdict, Counter
import re


def load_and_clean_data():
    """Load the car features data and clean it."""
    print("🔍 Loading car features data...")

    # Load the original dataset to get car model information
    original_df = pd.read_csv("facebook_ads_electric_vehicles_enhanced_with_gpt4.csv")
    print(f"✅ Loaded {len(original_df)} original records")

    # Load the JSON results which have the features
    with open("car_features_complete.json", "r", encoding="utf-8") as f:
        features_data = json.load(f)
    print(f"✅ Loaded {len(features_data)} feature analysis records")

    # Create a dataframe from features data
    features_list = []
    for i, item in enumerate(features_data):
        if i < len(original_df):  # Match with original data by index
            original_row = original_df.iloc[i]
            features_list.append(
                {
                    "ad_archive_id": original_row.get("ad_archive_id", ""),
                    "page_name": original_row.get("page_name", ""),
                    "matched_car_models": original_row.get("matched_car_models", ""),
                    "page_classification": original_row.get("page_classification", ""),
                    "car_features": item.get("car_features"),
                    "features_found": (
                        "Yes" if item.get("car_features") is not None else "No"
                    ),
                    "success": item.get("success", False),
                }
            )

    df = pd.DataFrame(features_list)
    print(f"✅ Created merged dataset with {len(df)} records")

    # Filter for records with features found
    features_df = df[df["features_found"] == "Yes"].copy()
    print(f"📊 Found {len(features_df)} ads with car features")

    return df, features_df


def extract_feature_categories(feature_text):
    """Extract and categorize features from the text."""
    if pd.isna(feature_text) or feature_text is None:
        return {}

    categories = {
        "Range_Charging": [],
        "Performance": [],
        "Interior_Comfort": [],
        "Infotainment_Audio": [],
        "Exterior_Design": [],
        "Safety_Assistance": [],
        "Connectivity_Digital": [],
        "Electric_Drivetrain": [],
        "Other": [],
    }

    # Convert to string and split by lines
    text = str(feature_text).lower()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines:
        # Remove bullet points and dashes
        clean_line = re.sub(r"^[-•*]\s*", "", line).strip()

        if not clean_line:
            continue

        # Categorize features based on keywords
        if any(
            keyword in clean_line
            for keyword in ["range", "charging", "battery", "kwh", "miles", "km"]
        ):
            categories["Range_Charging"].append(clean_line)
        elif any(
            keyword in clean_line
            for keyword in [
                "performance",
                "acceleration",
                "horsepower",
                "torque",
                "speed",
                "power",
            ]
        ):
            categories["Performance"].append(clean_line)
        elif any(
            keyword in clean_line
            for keyword in [
                "interior",
                "seat",
                "comfort",
                "space",
                "cargo",
                "luxury",
                "leather",
            ]
        ):
            categories["Interior_Comfort"].append(clean_line)
        elif any(
            keyword in clean_line
            for keyword in [
                "infotainment",
                "audio",
                "screen",
                "display",
                "carplay",
                "android",
                "navigation",
                "bluetooth",
            ]
        ):
            categories["Infotainment_Audio"].append(clean_line)
        elif any(
            keyword in clean_line
            for keyword in [
                "exterior",
                "design",
                "led",
                "lights",
                "wheels",
                "styling",
                "aerodynamic",
            ]
        ):
            categories["Exterior_Design"].append(clean_line)
        elif any(
            keyword in clean_line
            for keyword in [
                "safety",
                "assist",
                "autopilot",
                "collision",
                "lane",
                "parking",
                "camera",
                "sensor",
            ]
        ):
            categories["Safety_Assistance"].append(clean_line)
        elif any(
            keyword in clean_line
            for keyword in [
                "connectivity",
                "digital",
                "app",
                "wireless",
                "smart",
                "remote",
                "update",
            ]
        ):
            categories["Connectivity_Digital"].append(clean_line)
        elif any(
            keyword in clean_line
            for keyword in ["electric", "ηλεκτρικό", "ev", "drivetrain", "motor"]
        ):
            categories["Electric_Drivetrain"].append(clean_line)
        else:
            categories["Other"].append(clean_line)

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def analyze_features_by_model(features_df):
    """Analyze features by car model."""
    print("\n🚗 Analyzing features by car model...")

    model_features = defaultdict(lambda: defaultdict(list))
    model_counts = defaultdict(int)

    for _, row in features_df.iterrows():
        models = (
            str(row["matched_car_models"]).split(";")
            if pd.notna(row["matched_car_models"])
            else ["Unknown"]
        )
        features = extract_feature_categories(row["car_features"])

        for model in models:
            model = model.strip()
            model_counts[model] += 1

            for category, feature_list in features.items():
                model_features[model][category].extend(feature_list)

    return model_features, model_counts


def create_feature_frequency_analysis(model_features):
    """Create frequency analysis of features."""
    print("\n📈 Creating feature frequency analysis...")

    all_features = defaultdict(int)
    category_totals = defaultdict(int)

    for model, categories in model_features.items():
        for category, features in categories.items():
            category_totals[category] += len(features)
            for feature in features:
                all_features[feature] += 1

    return all_features, category_totals


def generate_report(
    df, features_df, model_features, model_counts, all_features, category_totals
):
    """Generate comprehensive analysis report."""

    print("\n" + "=" * 80)
    print("🚗⚡ CAR FEATURES ANALYSIS REPORT")
    print("=" * 80)

    # Overall Statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total ads analyzed: {len(df):,}")
    print(f"   Ads with car features: {len(features_df):,}")
    print(f"   Ads without features: {len(df) - len(features_df):,}")
    print(f"   Feature extraction rate: {len(features_df)/len(df)*100:.1f}%")

    # Top Car Models with Features
    print(f"\n🏆 TOP CAR MODELS BY FEATURE MENTIONS:")
    sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (model, count) in enumerate(sorted_models[:15], 1):
        print(f"   {i:2d}. {model:<25} {count:3d} ads with features")

    # Feature Categories Analysis
    print(f"\n📋 FEATURE CATEGORIES ANALYSIS:")
    sorted_categories = sorted(
        category_totals.items(), key=lambda x: x[1], reverse=True
    )
    for category, count in sorted_categories:
        category_name = category.replace("_", " & ")
        print(f"   {category_name:<25} {count:3d} mentions")

    # Most Common Features
    print(f"\n🔥 MOST FREQUENTLY MENTIONED FEATURES:")
    sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, count) in enumerate(sorted_features[:20], 1):
        if len(feature) > 60:
            feature = feature[:57] + "..."
        print(f"   {i:2d}. {feature:<60} {count:2d} times")

    # Model-Specific Analysis
    print(f"\n🚗 DETAILED MODEL ANALYSIS:")
    top_models = [model for model, _ in sorted_models[:10]]

    for model in top_models:
        if (
            model in model_features and model_counts[model] >= 5
        ):  # Only show models with 5+ feature mentions
            print(f"\n   📌 {model} ({model_counts[model]} ads with features):")
            model_cats = model_features[model]

            for category, features in model_cats.items():
                if features:  # Only show categories with features
                    category_name = category.replace("_", " & ")
                    feature_count = len(features)
                    unique_features = len(set(features))
                    print(
                        f"      • {category_name}: {feature_count} mentions ({unique_features} unique)"
                    )

                    # Show top features for this model/category
                    feature_counter = Counter(features)
                    top_features = feature_counter.most_common(3)
                    for feature, count in top_features:
                        if len(feature) > 50:
                            feature = feature[:47] + "..."
                        print(f"        - {feature} ({count}x)")

    print(f"\n" + "=" * 80)
    print("📁 Analysis complete! Check the generated CSV files for detailed data.")
    print("=" * 80)


def save_detailed_analysis(model_features, model_counts):
    """Save detailed analysis to CSV files."""
    print("\n💾 Saving detailed analysis files...")

    # Model summary
    model_summary = []
    for model, count in model_counts.items():
        categories = (
            list(model_features[model].keys()) if model in model_features else []
        )
        total_features = (
            sum(len(features) for features in model_features[model].values())
            if model in model_features
            else 0
        )

        model_summary.append(
            {
                "car_model": model,
                "ads_with_features": count,
                "total_feature_mentions": total_features,
                "feature_categories": len(categories),
                "categories_mentioned": "; ".join(categories),
            }
        )

    model_df = pd.DataFrame(model_summary)
    model_df = model_df.sort_values("ads_with_features", ascending=False)
    model_df.to_csv("car_features_by_model_summary.csv", index=False)
    print("   ✅ Saved: car_features_by_model_summary.csv")

    # Detailed feature breakdown
    detailed_breakdown = []
    for model, categories in model_features.items():
        for category, features in categories.items():
            feature_counter = Counter(features)
            for feature, count in feature_counter.items():
                detailed_breakdown.append(
                    {
                        "car_model": model,
                        "feature_category": category.replace("_", " & "),
                        "feature_description": feature,
                        "mention_count": count,
                    }
                )

    detailed_df = pd.DataFrame(detailed_breakdown)
    detailed_df = detailed_df.sort_values(
        ["car_model", "feature_category", "mention_count"],
        ascending=[True, True, False],
    )
    detailed_df.to_csv("car_features_detailed_breakdown.csv", index=False)
    print("   ✅ Saved: car_features_detailed_breakdown.csv")


def main():
    """Main analysis function."""
    print("🚗🔍 CAR FEATURES ANALYSIS")
    print("=" * 50)

    # Load data
    df, features_df = load_and_clean_data()

    # Analyze features by model
    model_features, model_counts = analyze_features_by_model(features_df)

    # Create frequency analysis
    all_features, category_totals = create_feature_frequency_analysis(model_features)

    # Generate report
    generate_report(
        df, features_df, model_features, model_counts, all_features, category_totals
    )

    # Save detailed analysis
    save_detailed_analysis(model_features, model_counts)


if __name__ == "__main__":
    main()
