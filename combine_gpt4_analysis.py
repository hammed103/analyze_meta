#!/usr/bin/env python3
"""
Combine GPT-4 image analysis results with the main EV ads dataset.
Creates an enhanced dataset with AI-powered text extraction and theme analysis.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime


def load_and_validate_data():
    """Load both datasets and validate them."""
    print("🔄 Loading datasets...")
    
    # Load main dataset
    try:
        main_df = pd.read_csv('facebook_ads_electric_vehicles_with_classifications.csv')
        print(f"✅ Main dataset loaded: {len(main_df)} records")
    except FileNotFoundError:
        print("❌ Main dataset not found: facebook_ads_electric_vehicles_with_classifications.csv")
        return None, None
    
    # Load GPT-4 analysis
    try:
        gpt4_df = pd.read_csv('gpt4_analysis_summary.csv')
        print(f"✅ GPT-4 analysis loaded: {len(gpt4_df)} records")
    except FileNotFoundError:
        print("❌ GPT-4 analysis not found: gpt4_analysis_summary.csv")
        return None, None
    
    return main_df, gpt4_df


def clean_and_prepare_gpt4_data(gpt4_df):
    """Clean and prepare GPT-4 analysis data for merging."""
    print("🧹 Cleaning GPT-4 analysis data...")
    
    # Remove failed analyses
    successful_analyses = gpt4_df[gpt4_df['gpt4_text_analysis'].notna()].copy()
    failed_count = len(gpt4_df) - len(successful_analyses)
    
    if failed_count > 0:
        print(f"⚠️  Removed {failed_count} failed analyses")
    
    print(f"✅ {len(successful_analyses)} successful analyses to merge")
    
    # Parse the text analysis to extract structured information
    successful_analyses = parse_gpt4_analysis(successful_analyses)
    
    return successful_analyses


def parse_gpt4_analysis(df):
    """Parse GPT-4 text analysis into structured fields."""
    print("📝 Parsing GPT-4 text analysis...")
    
    # Initialize new columns
    df['extracted_text'] = ''
    df['ad_theme'] = ''
    df['text_extraction_success'] = False
    
    for idx, row in df.iterrows():
        analysis = str(row['gpt4_text_analysis'])
        
        try:
            # Split the analysis into TEXT FOUND and THEME sections
            if 'TEXT FOUND:' in analysis and 'THEME:' in analysis:
                parts = analysis.split('THEME:')
                
                # Extract text section
                text_part = parts[0].replace('TEXT FOUND:', '').strip()
                # Clean up the text (remove bullet points, extra whitespace)
                text_lines = [line.strip('- ').strip() for line in text_part.split('\n') if line.strip()]
                extracted_text = ' | '.join([line for line in text_lines if line])
                
                # Extract theme section
                theme_part = parts[1].strip() if len(parts) > 1 else ''
                
                df.at[idx, 'extracted_text'] = extracted_text
                df.at[idx, 'ad_theme'] = theme_part
                df.at[idx, 'text_extraction_success'] = True
            else:
                # Fallback: use the entire analysis as theme if structure is different
                df.at[idx, 'ad_theme'] = analysis
                df.at[idx, 'text_extraction_success'] = False
                
        except Exception as e:
            print(f"⚠️  Error parsing analysis for row {idx}: {e}")
            df.at[idx, 'text_extraction_success'] = False
    
    success_count = df['text_extraction_success'].sum()
    print(f"✅ Successfully parsed {success_count}/{len(df)} analyses")
    
    return df


def merge_datasets(main_df, gpt4_df):
    """Merge the main dataset with GPT-4 analysis results."""
    print("🔗 Merging datasets...")
    
    # Prepare merge keys
    # Use image URL as the primary merge key since it's most reliable
    main_df['merge_key'] = main_df['first_image_url']
    gpt4_df['merge_key'] = gpt4_df['image_url']
    
    # Perform left join to keep all main dataset records
    merged_df = main_df.merge(
        gpt4_df[['merge_key', 'gpt4_text_analysis', 'extracted_text', 'ad_theme', 'text_extraction_success']], 
        on='merge_key', 
        how='left'
    )
    
    # Clean up
    merged_df = merged_df.drop('merge_key', axis=1)
    
    # Fill NaN values for records without GPT-4 analysis
    merged_df['gpt4_text_analysis'] = merged_df['gpt4_text_analysis'].fillna('')
    merged_df['extracted_text'] = merged_df['extracted_text'].fillna('')
    merged_df['ad_theme'] = merged_df['ad_theme'].fillna('')
    merged_df['text_extraction_success'] = merged_df['text_extraction_success'].fillna(False)
    
    # Add analysis metadata
    merged_df['has_gpt4_analysis'] = merged_df['gpt4_text_analysis'] != ''
    merged_df['analysis_date'] = datetime.now().strftime('%Y-%m-%d')
    
    print(f"✅ Merged dataset created: {len(merged_df)} total records")
    print(f"📊 Records with GPT-4 analysis: {merged_df['has_gpt4_analysis'].sum()}")
    print(f"📊 Records with successful text extraction: {merged_df['text_extraction_success'].sum()}")
    
    return merged_df


def create_analysis_summary(merged_df):
    """Create a summary of the analysis results."""
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 50)
    
    total_records = len(merged_df)
    with_images = merged_df['first_image_url'].notna().sum()
    with_gpt4 = merged_df['has_gpt4_analysis'].sum()
    successful_extraction = merged_df['text_extraction_success'].sum()
    
    print(f"📈 Total ad records: {total_records:,}")
    print(f"📸 Records with images: {with_images:,} ({with_images/total_records*100:.1f}%)")
    print(f"🤖 Records with GPT-4 analysis: {with_gpt4:,} ({with_gpt4/total_records*100:.1f}%)")
    print(f"✅ Successful text extractions: {successful_extraction:,} ({successful_extraction/total_records*100:.1f}%)")
    
    # Analysis by car model
    print(f"\n🚗 GPT-4 Analysis by Car Model:")
    model_analysis = merged_df[merged_df['has_gpt4_analysis']]['matched_car_models'].value_counts().head(10)
    for model, count in model_analysis.items():
        print(f"   {model}: {count} analyzed ads")
    
    # Analysis by advertiser type
    if 'page_classification' in merged_df.columns:
        print(f"\n🏢 GPT-4 Analysis by Advertiser Type:")
        type_analysis = merged_df[merged_df['has_gpt4_analysis']]['page_classification'].value_counts()
        for adv_type, count in type_analysis.items():
            print(f"   {adv_type}: {count} analyzed ads")
    
    # Sample extracted text
    print(f"\n📝 Sample Extracted Text:")
    sample_extractions = merged_df[merged_df['text_extraction_success']]['extracted_text'].head(3)
    for i, text in enumerate(sample_extractions, 1):
        print(f"   {i}. {text[:100]}...")


def save_enhanced_dataset(merged_df):
    """Save the enhanced dataset with multiple formats."""
    print("\n💾 Saving enhanced dataset...")
    
    # Save main enhanced CSV
    output_file = 'facebook_ads_electric_vehicles_enhanced_with_gpt4.csv'
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ Enhanced dataset saved: {output_file}")
    
    # Save analysis-only subset for quick access
    analysis_subset = merged_df[merged_df['has_gpt4_analysis']].copy()
    analysis_file = 'ev_ads_with_gpt4_analysis_only.csv'
    analysis_subset.to_csv(analysis_file, index=False, encoding='utf-8')
    print(f"✅ Analysis subset saved: {analysis_file}")
    
    # Save summary statistics
    summary_stats = {
        'total_records': len(merged_df),
        'records_with_images': merged_df['first_image_url'].notna().sum(),
        'records_with_gpt4_analysis': merged_df['has_gpt4_analysis'].sum(),
        'successful_text_extractions': merged_df['text_extraction_success'].sum(),
        'analysis_date': datetime.now().isoformat(),
        'car_model_breakdown': merged_df[merged_df['has_gpt4_analysis']]['matched_car_models'].value_counts().to_dict(),
    }
    
    if 'page_classification' in merged_df.columns:
        summary_stats['advertiser_type_breakdown'] = merged_df[merged_df['has_gpt4_analysis']]['page_classification'].value_counts().to_dict()
    
    with open('gpt4_analysis_summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✅ Summary statistics saved: gpt4_analysis_summary_stats.json")
    
    return output_file


def main():
    """Main function to combine datasets."""
    print("🚗⚡ COMBINING GPT-4 ANALYSIS WITH MAIN DATASET")
    print("=" * 60)
    
    # Load data
    main_df, gpt4_df = load_and_validate_data()
    if main_df is None or gpt4_df is None:
        return
    
    # Clean GPT-4 data
    gpt4_clean = clean_and_prepare_gpt4_data(gpt4_df)
    
    # Merge datasets
    merged_df = merge_datasets(main_df, gpt4_clean)
    
    # Create summary
    create_analysis_summary(merged_df)
    
    # Save results
    output_file = save_enhanced_dataset(merged_df)
    
    print(f"\n🎉 COMBINATION COMPLETE!")
    print(f"📁 Enhanced dataset: {output_file}")
    print(f"🔗 Ready for dashboard integration!")
    print(f"\nNext steps:")
    print(f"1. Update your dashboard to use the enhanced dataset")
    print(f"2. Add GPT-4 analysis visualizations")
    print(f"3. Explore text extraction and theme patterns")


if __name__ == "__main__":
    main()
