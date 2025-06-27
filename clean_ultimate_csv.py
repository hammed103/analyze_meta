#!/usr/bin/env python3
"""
Clean the ultimate AI enhanced CSV by removing unused columns to reduce file size.
"""

import pandas as pd
import os

def main():
    input_file = "facebook_ads_electric_vehicles_ultimate_ai_enhanced.csv"
    output_file = "facebook_ads_electric_vehicles_clean.csv"
    
    print(f"Loading data from {input_file}...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    # Load the data
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")
    print(f"Original file size: {os.path.getsize(input_file) / (1024*1024):.1f} MB")
    
    # Define columns that are actually used in the dashboard
    used_columns = [
        'page_name',           # Advertiser names and analysis
        'ad_title',            # Displayed in ad details
        'ad_text',             # Used for keyword analysis and display
        'cta_text',            # CTA analysis charts
        'display_format',      # Display format analysis
        'start_date',          # Timeline analysis and display
        'first_image_url',     # Image gallery (essential!)
        'male_percentage',     # Gender targeting analysis
        'female_percentage',   # Gender targeting analysis
        'total_male_audience', # Scatter plot sizing
        'total_female_audience', # Metrics
        'page_like_count',     # Correlation analysis
        'matched_car_models',  # Core filtering and analysis
        'page_classification', # Advertiser type analysis
        'openai_summary',      # AI analysis display
        'ad_theme',            # Image theme display
        'targeted_countries',  # Country analysis
    ]
    
    # Check which columns exist in the dataframe
    available_columns = [col for col in used_columns if col in df.columns]
    missing_columns = [col for col in used_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    print(f"Keeping {len(available_columns)} out of {len(df.columns)} columns")
    print(f"Columns to keep: {available_columns}")
    
    # Show columns being removed
    removed_columns = [col for col in df.columns if col not in available_columns]
    print(f"\nRemoving {len(removed_columns)} unused columns:")
    for i, col in enumerate(removed_columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Create cleaned dataframe
    df_clean = df[available_columns].copy()
    
    print(f"\nCleaned data shape: {df_clean.shape}")
    
    # Save the cleaned data
    print(f"Saving cleaned data to {output_file}...")
    df_clean.to_csv(output_file, index=False)
    
    # Show file size comparison
    new_size = os.path.getsize(output_file) / (1024*1024)
    original_size = os.path.getsize(input_file) / (1024*1024)
    reduction = ((original_size - new_size) / original_size) * 100
    
    print(f"\n✅ Success!")
    print(f"Original file: {original_size:.1f} MB")
    print(f"Cleaned file: {new_size:.1f} MB")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"Saved {original_size - new_size:.1f} MB of space!")
    
    # Show sample of cleaned data
    print(f"\nSample of cleaned data:")
    print(df_clean.head(2))

if __name__ == "__main__":
    main()
