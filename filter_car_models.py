#!/usr/bin/env python3
"""
Filter Facebook Ads CSV to only include ads mentioning specific car models.
Searches across all columns for brand and model mentions.
"""

import pandas as pd
import re
import sys
from typing import List, Set


def create_search_patterns(car_models: List[str]) -> List[str]:
    """
    Create regex patterns for flexible matching of car models.
    Handles variations in spacing, punctuation, and case.
    """
    patterns = []

    for model in car_models:
        # Clean the model name and create variations
        clean_model = model.strip()

        # Create pattern that handles:
        # - Case insensitive matching
        # - Optional spaces, dots, hyphens
        # - Word boundaries to avoid partial matches

        # Replace spaces and dots with flexible pattern
        pattern = re.escape(clean_model)
        pattern = pattern.replace(
            r"\ ", r"[\s\-\.]*"
        )  # Space can be space, dash, or dot
        pattern = pattern.replace(r"\.", r"[\.\s\-]*")  # Dot can be dot, space, or dash
        pattern = pattern.replace(
            r"\-", r"[\-\s\.]*"
        )  # Dash can be dash, space, or dot

        # Add word boundaries and case insensitive flag
        pattern = r"\b" + pattern + r"\b"
        patterns.append(pattern)

        # Also add exact match pattern
        exact_pattern = r"\b" + re.escape(clean_model) + r"\b"
        if exact_pattern not in patterns:
            patterns.append(exact_pattern)

    return patterns


def search_in_text(text: str, patterns: List[str]) -> bool:
    """
    Search for any of the patterns in the given text.
    Returns True if any pattern is found.
    """
    if pd.isna(text) or text == "":
        return False

    text_str = str(text).lower()

    for pattern in patterns:
        if re.search(pattern, text_str, re.IGNORECASE):
            return True

    return False


def filter_car_ads(input_csv: str, output_csv: str, car_models: List[str]):
    """
    Filter the CSV to only include rows mentioning the specified car models.
    """
    print(f"Loading CSV data from {input_csv}...")

    try:
        # Read the CSV
        df = pd.read_csv(input_csv, low_memory=False)
        print(f"Loaded {len(df)} total records with {len(df.columns)} columns")

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Create search patterns
    print(f"Creating search patterns for {len(car_models)} car models...")
    patterns = create_search_patterns(car_models)

    print("Car models to search for:")
    for i, model in enumerate(car_models, 1):
        print(f"  {i:2d}. {model}")

    # Create a boolean mask for rows that match
    print("\nSearching across all columns...")
    matching_rows = pd.Series([False] * len(df))

    # Search in all columns
    for col_idx, column in enumerate(df.columns):
        if col_idx % 10 == 0:
            print(f"  Searching column {col_idx + 1}/{len(df.columns)}: {column}")

        # Check each cell in this column
        column_matches = df[column].apply(lambda x: search_in_text(x, patterns))
        matching_rows = matching_rows | column_matches

    # Filter the dataframe
    filtered_df = df[matching_rows].copy()

    print(f"\nFiltering results:")
    print(f"  Original records: {len(df)}")
    print(f"  Matching records: {len(filtered_df)}")
    print(f"  Filtered out: {len(df) - len(filtered_df)}")
    print(f"  Match rate: {len(filtered_df)/len(df)*100:.2f}%")

    if len(filtered_df) == 0:
        print("\nNo matching records found!")
        print("This could mean:")
        print("- The car models are not mentioned in the ads")
        print("- The model names are written differently")
        print("- The search patterns need adjustment")
        return

    # Add a column showing which models were found
    def find_matching_models(row):
        matches = []
        row_text = " ".join([str(val) for val in row.values if pd.notna(val)])

        for model in car_models:
            model_patterns = create_search_patterns([model])
            if any(
                re.search(pattern, row_text, re.IGNORECASE)
                for pattern in model_patterns
            ):
                matches.append(model)

        return "; ".join(matches)

    print("Adding matched models column...")
    filtered_df["matched_car_models"] = filtered_df.apply(find_matching_models, axis=1)

    # Save the filtered data
    print(f"\nSaving filtered data to {output_csv}...")
    try:
        filtered_df.to_csv(output_csv, index=False)

        # Calculate file size
        import os

        size_mb = os.path.getsize(output_csv) / (1024 * 1024)
        print(f"Successfully saved {len(filtered_df)} records to {output_csv}")
        print(f"File size: {size_mb:.1f} MB")

        # Show summary of matches by model
        print(f"\nMatches by car model:")
        model_counts = {}
        for _, row in filtered_df.iterrows():
            matched_models = (
                row["matched_car_models"].split("; ")
                if row["matched_car_models"]
                else []
            )
            for model in matched_models:
                model_counts[model] = model_counts.get(model, 0) + 1

        for model, count in sorted(
            model_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {model}: {count} ads")

        # Show sample of filtered data
        print(f"\nSample of filtered data (first 3 records):")
        sample_cols = ["page_name", "ad_title", "ad_text", "matched_car_models"]
        available_cols = [col for col in sample_cols if col in filtered_df.columns]

        for i, (_, row) in enumerate(filtered_df.head(3).iterrows()):
            print(f"\nRecord {i+1}:")
            for col in available_cols:
                value = (
                    str(row[col])[:100] + "..."
                    if len(str(row[col])) > 100
                    else str(row[col])
                )
                print(f"  {col}: {value}")

    except Exception as e:
        print(f"Error saving filtered CSV: {e}")


def main():
    # Define the car models to search for
    car_models = [
        "Cupra Born",
        "Mini Aceman E",
        "Volkswagen ID.3",
        "Kia EV3",
        "Volvo EX30",
        "Renault Megane E-Tech",
        "Volvo EC40",
        "Volkswagen ID.5",
        "MG4",
        "BMW iX2",
        "Hyundai Ioniq 5",
        "Volkswagen ID.4",
        "Audi Q4 e-tron",
        "BMW iX1",
        "BMW iX3",
        "Tesla Model Y",
    ]

    # Default file names
    input_file = "facebook_ads_clean.csv"
    output_file = "facebook_ads_electric_vehicles.csv"

    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    print("=== FACEBOOK ADS CAR MODEL FILTER ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Run the filtering
    filter_car_ads(input_file, output_file, car_models)

    print(f"\n=== FILTERING COMPLETE ===")
    print(f"Filtered data saved to: {output_file}")


if __name__ == "__main__":
    main()
