#!/usr/bin/env python3
"""
Create a small sample dataset that fits GitHub's 100MB file size limit.
"""

import pandas as pd
import os

def main():
    input_file = "facebook_ads_electric_vehicles_clean.csv"
    output_file = "facebook_ads_electric_vehicles_sample.csv"
    
    print(f"Loading data from {input_file}...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    # Load the data
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")
    print(f"Original file size: {os.path.getsize(input_file) / (1024*1024):.1f} MB")
    
    # Strategy 1: Take a representative sample
    # Sample 5000 rows (about 10% of the data) to get under 25MB
    sample_size = 5000
    
    print(f"\nCreating sample with {sample_size:,} rows...")
    
    # Stratified sampling to ensure we get variety across car models
    if 'matched_car_models' in df.columns:
        # Get proportional sample from each car model
        df_sample = df.groupby('matched_car_models', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df)))))
        ).reset_index(drop=True)
        
        # If we don't have enough, fill with random sample
        if len(df_sample) < sample_size:
            remaining = sample_size - len(df_sample)
            additional = df[~df.index.isin(df_sample.index)].sample(min(remaining, len(df) - len(df_sample)))
            df_sample = pd.concat([df_sample, additional]).reset_index(drop=True)
    else:
        # Simple random sample
        df_sample = df.sample(min(sample_size, len(df))).reset_index(drop=True)
    
    # Strategy 2: Truncate long text fields to reduce size further
    text_columns = ['ad_text', 'openai_summary']
    for col in text_columns:
        if col in df_sample.columns:
            # Truncate to reasonable lengths
            if col == 'ad_text':
                df_sample[col] = df_sample[col].astype(str).str[:500]  # 500 chars max
            elif col == 'openai_summary':
                df_sample[col] = df_sample[col].astype(str).str[:400]  # 400 chars max
    
    print(f"Sample data shape: {df_sample.shape}")
    
    # Save the sample
    print(f"Saving sample data to {output_file}...")
    df_sample.to_csv(output_file, index=False)
    
    # Show file size comparison
    new_size = os.path.getsize(output_file) / (1024*1024)
    original_size = os.path.getsize(input_file) / (1024*1024)
    reduction = ((original_size - new_size) / original_size) * 100
    
    print(f"\n✅ Success!")
    print(f"Original file: {original_size:.1f} MB ({len(df):,} rows)")
    print(f"Sample file: {new_size:.1f} MB ({len(df_sample):,} rows)")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"Saved {original_size - new_size:.1f} MB of space!")
    
    # Check if it fits GitHub limits
    if new_size < 100:
        print(f"✅ File size ({new_size:.1f} MB) is under GitHub's 100MB limit!")
    else:
        print(f"⚠️  File size ({new_size:.1f} MB) is still over GitHub's 100MB limit")
        print("Consider reducing sample size further or truncating text more aggressively")
    
    # Show sample distribution
    if 'matched_car_models' in df_sample.columns:
        print(f"\nSample distribution by car model:")
        model_counts = df_sample['matched_car_models'].value_counts().head(10)
        for model, count in model_counts.items():
            print(f"  {model}: {count} ads")

if __name__ == "__main__":
    main()
