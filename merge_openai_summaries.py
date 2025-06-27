#!/usr/bin/env python3
"""
Merge OpenAI summaries with the main EV ads dataset.
Creates a comprehensive dataset with AI-powered summaries.
"""

import pandas as pd
import numpy as np
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
    
    # Load OpenAI summaries
    try:
        openai_df = pd.read_csv('openai_ad_summary.csv')
        print(f"✅ OpenAI summaries loaded: {len(openai_df)} records")
    except FileNotFoundError:
        print("❌ OpenAI summaries not found: openai_ad_summary.csv")
        return None, None
    
    return main_df, openai_df


def merge_datasets(main_df, openai_df):
    """Merge the main dataset with OpenAI summaries."""
    print("🔗 Merging datasets...")
    
    # Use ad_archive_id as the primary merge key
    merged_df = main_df.merge(
        openai_df[['ad_archive_id', 'openai_summary']], 
        on='ad_archive_id', 
        how='left'
    )
    
    # Fill NaN values for records without OpenAI analysis
    merged_df['openai_summary'] = merged_df['openai_summary'].fillna('')
    
    # Add analysis metadata
    merged_df['has_openai_summary'] = merged_df['openai_summary'] != ''
    merged_df['summary_date'] = datetime.now().strftime('%Y-%m-%d')
    
    print(f"✅ Merged dataset created: {len(merged_df)} total records")
    print(f"📊 Records with OpenAI summaries: {merged_df['has_openai_summary'].sum()}")
    
    return merged_df


def create_analysis_summary(merged_df):
    """Create a summary of the merged results."""
    print("\n📊 MERGE SUMMARY")
    print("=" * 50)
    
    total_records = len(merged_df)
    with_summaries = merged_df['has_openai_summary'].sum()
    
    print(f"📈 Total ad records: {total_records:,}")
    print(f"🤖 Records with OpenAI summaries: {with_summaries:,} ({with_summaries/total_records*100:.1f}%)")
    
    # Analysis by car model
    print(f"\n🚗 OpenAI Summaries by Car Model:")
    model_analysis = merged_df[merged_df['has_openai_summary']]['matched_car_models'].value_counts().head(10)
    for model, count in model_analysis.items():
        print(f"   {model}: {count} summaries")
    
    # Analysis by advertiser type
    if 'page_classification' in merged_df.columns:
        print(f"\n🏢 OpenAI Summaries by Advertiser Type:")
        type_analysis = merged_df[merged_df['has_openai_summary']]['page_classification'].value_counts()
        for adv_type, count in type_analysis.items():
            print(f"   {adv_type}: {count} summaries")
    
    # Sample summary
    print(f"\n📝 Sample OpenAI Summary:")
    sample = merged_df[merged_df['has_openai_summary']].iloc[0]
    print(f"Advertiser: {sample['page_name']}")
    print(f"Model: {sample['matched_car_models']}")
    print(f"Summary: {sample['openai_summary'][:200]}...")


def save_merged_dataset(merged_df):
    """Save the merged dataset."""
    print("\n💾 Saving merged dataset...")
    
    # Save main merged CSV
    output_file = 'facebook_ads_electric_vehicles_with_openai_summaries.csv'
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ Merged dataset saved: {output_file}")
    
    # Save summaries-only subset for quick access
    summaries_subset = merged_df[merged_df['has_openai_summary']].copy()
    summaries_file = 'ev_ads_with_openai_summaries_only.csv'
    summaries_subset.to_csv(summaries_file, index=False, encoding='utf-8')
    print(f"✅ Summaries subset saved: {summaries_file}")
    
    return output_file


def create_summary_stats(merged_df):
    """Create summary statistics."""
    print("\n📈 Creating summary statistics...")
    
    stats = {
        'merge_date': datetime.now().isoformat(),
        'total_records': len(merged_df),
        'records_with_openai_summaries': merged_df['has_openai_summary'].sum(),
        'coverage_percentage': (merged_df['has_openai_summary'].sum() / len(merged_df)) * 100,
        'car_model_breakdown': merged_df[merged_df['has_openai_summary']]['matched_car_models'].value_counts().head(10).to_dict(),
    }
    
    if 'page_classification' in merged_df.columns:
        stats['advertiser_type_breakdown'] = merged_df[merged_df['has_openai_summary']]['page_classification'].value_counts().to_dict()
    
    # Save stats
    import json
    with open('openai_merge_summary_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Summary statistics saved: openai_merge_summary_stats.json")


def analyze_summary_content(merged_df):
    """Analyze the content of OpenAI summaries."""
    print("\n🔍 ANALYZING SUMMARY CONTENT")
    print("=" * 50)
    
    summaries_df = merged_df[merged_df['has_openai_summary']].copy()
    
    # Count mentions of feature categories
    feature_categories = [
        'Range and Charging',
        'Performance', 
        'Interior and Comfort',
        'Infotainment & Audio',
        'Exterior Design',
        'Safety & Assistance',
        'Connectivity and Digital'
    ]
    
    print("📊 Feature Category Mentions in Summaries:")
    for category in feature_categories:
        count = summaries_df['openai_summary'].str.contains(category, case=False, na=False).sum()
        percentage = (count / len(summaries_df)) * 100
        print(f"   {category}: {count} ads ({percentage:.1f}%)")
    
    # Analyze by car model
    print(f"\n🚗 Top Models with Feature-Rich Summaries:")
    for model in summaries_df['matched_car_models'].value_counts().head(5).index:
        model_summaries = summaries_df[summaries_df['matched_car_models'] == model]
        feature_mentions = 0
        for category in feature_categories:
            feature_mentions += model_summaries['openai_summary'].str.contains(category, case=False, na=False).sum()
        
        avg_features = feature_mentions / len(model_summaries) if len(model_summaries) > 0 else 0
        print(f"   {model}: {len(model_summaries)} ads, {avg_features:.1f} avg features mentioned")


def main():
    """Main function to merge datasets."""
    print("🚗⚡ MERGING OPENAI SUMMARIES WITH MAIN DATASET")
    print("=" * 60)
    
    # Load data
    main_df, openai_df = load_and_validate_data()
    if main_df is None or openai_df is None:
        return
    
    # Merge datasets
    merged_df = merge_datasets(main_df, openai_df)
    
    # Create summary
    create_analysis_summary(merged_df)
    
    # Analyze summary content
    analyze_summary_content(merged_df)
    
    # Save results
    output_file = save_merged_dataset(merged_df)
    
    # Create summary statistics
    create_summary_stats(merged_df)
    
    print(f"\n🎉 MERGE COMPLETE!")
    print(f"📁 Enhanced dataset: {output_file}")
    print(f"🔗 Ready for dashboard integration!")
    print(f"\nNext steps:")
    print(f"1. Update your dashboard to use the enhanced dataset")
    print(f"2. Add OpenAI summary visualizations")
    print(f"3. Create search functionality for summaries")
    print(f"4. Analyze feature patterns across brands/models")


if __name__ == "__main__":
    main()
