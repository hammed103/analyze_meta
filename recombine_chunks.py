#!/usr/bin/env python3
"""
Recombine CSV chunks back into the original large file
Auto-generated script
"""

import pandas as pd
import glob
import os

def recombine_chunks():
    """Recombine all chunk files into the original file."""
    data_dir = "Data"
    chunk_pattern = os.path.join(data_dir, "facebook_ads_chunk_*.csv")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        print(f"No chunk files found in {data_dir} with pattern facebook_ads_chunk_*.csv")
        return None

    print(f"Found {len(chunk_files)} chunk files in {data_dir}")
    print("Recombining...")

    df_list = []
    for file in chunk_files:
        print(f"Reading {file}...")
        df_chunk = pd.read_csv(file)
        df_list.append(df_chunk)

    combined_df = pd.concat(df_list, ignore_index=True)

    output_file = "facebook_ads_electric_vehicles_with_openai_summaries.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"Recombined into {output_file}")
    print(f"Total rows: {len(combined_df)}")
    return combined_df

if __name__ == "__main__":
    recombine_chunks()
