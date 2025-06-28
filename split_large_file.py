#!/usr/bin/env python3
"""
Split large CSV file into smaller chunks for GitHub compatibility
"""

import pandas as pd
import os


def split_csv_file(
    input_file, chunk_size=5000, output_prefix="facebook_ads_chunk", output_dir="Data"
):
    """
    Split a large CSV file into smaller chunks in a specified directory.

    Args:
        input_file (str): Path to the large CSV file
        chunk_size (int): Number of rows per chunk
        output_prefix (str): Prefix for output files
        output_dir (str): Directory to store the chunks
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    total_rows = len(df)

    print(f"Total rows: {total_rows}")
    print(f"Splitting into chunks of {chunk_size} rows...")
    print(f"Output directory: {output_dir}")

    chunk_files = []
    for i, start in enumerate(range(0, total_rows, chunk_size)):
        end = min(start + chunk_size, total_rows)
        chunk = df[start:end]

        output_file = os.path.join(output_dir, f"{output_prefix}_{i+1:02d}.csv")
        chunk.to_csv(output_file, index=False)
        chunk_files.append(output_file)

        print(f"Created {output_file} with {len(chunk)} rows")

    print(f"\nCreated {len(chunk_files)} chunk files:")
    for file in chunk_files:
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"  {file}: {size_mb:.1f} MB")

    # Create a recombine script
    create_recombine_script(chunk_files, input_file, output_dir)


def create_recombine_script(chunk_files, original_filename, output_dir):
    """Create a script to recombine the chunks."""
    # Get the base pattern from the first chunk file
    first_chunk = os.path.basename(chunk_files[0])
    pattern_parts = first_chunk.split("_")[:-1]  # Remove the number part
    pattern_base = "_".join(pattern_parts)

    script_content = f'''#!/usr/bin/env python3
"""
Recombine CSV chunks back into the original large file
Auto-generated script
"""

import pandas as pd
import glob
import os

def recombine_chunks():
    """Recombine all chunk files into the original file."""
    data_dir = "{output_dir}"
    chunk_pattern = os.path.join(data_dir, "{pattern_base}_*.csv")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        print(f"No chunk files found in {{data_dir}} with pattern {pattern_base}_*.csv")
        return None

    print(f"Found {{len(chunk_files)}} chunk files in {{data_dir}}")
    print("Recombining...")

    df_list = []
    for file in chunk_files:
        print(f"Reading {{file}}...")
        df_chunk = pd.read_csv(file)
        df_list.append(df_chunk)

    combined_df = pd.concat(df_list, ignore_index=True)

    output_file = "{original_filename}"
    combined_df.to_csv(output_file, index=False)

    print(f"Recombined into {{output_file}}")
    print(f"Total rows: {{len(combined_df)}}")
    return combined_df

if __name__ == "__main__":
    recombine_chunks()
'''

    with open("recombine_chunks.py", "w") as f:
        f.write(script_content)

    print(f"\nCreated recombine_chunks.py to recreate {original_filename}")


if __name__ == "__main__":
    input_file = "facebook_ads_electric_vehicles_clean.csv"

    if os.path.exists(input_file):
        # Use smaller chunk size for better GitHub compatibility
        # 3000 rows should create ~8MB chunks which are well under GitHub limits
        split_csv_file(input_file, chunk_size=5000)
    else:
        print(f"File {input_file} not found!")
        print("Make sure the file is in the current directory.")
