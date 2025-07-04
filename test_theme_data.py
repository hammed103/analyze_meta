#!/usr/bin/env python3
"""
Test script to verify theme analysis data is loading correctly
"""

import pandas as pd
import os

def test_theme_files():
    """Test if theme analysis files exist and can be loaded"""
    
    theme_files = {
        'theme_frequency_overall.csv': 'Overall theme frequencies',
        'theme_frequency_by_model.csv': 'Theme frequencies by car model',
        'lightweight_nlp_keywords.csv': 'NLP-discovered keywords',
        'lightweight_nlp_model_themes.csv': 'NLP themes by model',
        'lightweight_nlp_themes.csv': 'NLP theme definitions'
    }
    
    print("🎨 Testing Theme Analysis Data")
    print("=" * 50)
    
    loaded_files = 0
    
    for filename, description in theme_files.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                print(f"✅ {filename}")
                print(f"   📊 {description}")
                print(f"   📈 {len(df)} rows, {len(df.columns)} columns")
                
                # Show sample data
                if len(df) > 0:
                    print(f"   🔍 Sample: {df.iloc[0].to_dict()}")
                print()
                
                loaded_files += 1
                
            except Exception as e:
                print(f"❌ {filename} - Error loading: {e}")
                print()
        else:
            print(f"⚠️  {filename} - File not found")
            print()
    
    print(f"📊 Summary: {loaded_files}/{len(theme_files)} files loaded successfully")
    
    if loaded_files == len(theme_files):
        print("🎉 All theme analysis files are ready!")
        print("🚀 The dashboard should display theme analysis correctly.")
    else:
        print("⚠️  Some theme analysis files are missing.")
        print("💡 Run these commands to generate missing files:")
        print("   python3 simple_theme_analysis.py")
        print("   python3 lightweight_nlp_analysis.py")
    
    return loaded_files

def test_visualizations():
    """Test if visualization files exist"""
    
    viz_files = [
        'theme_frequency_chart.png',
        'model_theme_heatmap.png', 
        'nlp_keywords_chart.png',
        'theme_comparison_summary.png'
    ]
    
    print("\n🖼️  Testing Visualization Files")
    print("=" * 50)
    
    found_viz = 0
    for filename in viz_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"✅ {filename} ({file_size:,} bytes)")
            found_viz += 1
        else:
            print(f"⚠️  {filename} - Not found")
    
    print(f"\n📊 Visualizations: {found_viz}/{len(viz_files)} files found")
    
    if found_viz > 0:
        print("🎨 Visualization files are available for display!")
    
    return found_viz

def main():
    print("🧪 Theme Analysis Data Test")
    print("=" * 60)
    
    # Test data files
    data_files = test_theme_files()
    
    # Test visualization files
    viz_files = test_visualizations()
    
    # Overall summary
    print("\n" + "=" * 60)
    print("🎯 FINAL SUMMARY")
    print("=" * 60)
    
    if data_files >= 3:  # At least 3 key files
        print("✅ Theme analysis data is ready for dashboard!")
        print("🚀 Run: streamlit run ev_ads_dashboard.py")
        print("📍 Navigate to the '🎨 Theme Analysis' tab")
        
        if viz_files > 0:
            print("🖼️  Visualization files are also available!")
        
    else:
        print("❌ Theme analysis data needs to be generated first")
        print("🔧 Run the analysis scripts:")
        print("   python3 simple_theme_analysis.py")
        print("   python3 lightweight_nlp_analysis.py")

if __name__ == "__main__":
    main()
