import pandas as pd
import numpy as np
from collections import Counter
import re

def load_data():
    """Load the car advertisement data"""
    try:
        df = pd.read_csv('Data/facebook_ads_electric_vehicles_with_openai_summaries_cached.csv')
        print(f"Loaded {len(df)} rows of data")
        
        # Filter for rows that have OpenAI summaries
        df_with_summaries = df[df['openai_summary'].notna()].copy()
        print(f"Found {len(df_with_summaries)} rows with OpenAI summaries")
        
        return df_with_summaries
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_predefined_themes():
    """Return predefined themes based on common car ad elements"""
    return [
        "Modern/Sleek", "Urban/City", "Performance/Sporty", "Sophisticated", 
        "Innovative/Tech", "Dynamic", "Eco-Friendly", "Luxury", 
        "Contemporary", "Minimalist", "Bold/Striking", "Futuristic",
        "Nature/Outdoor", "Family-Oriented", "Professional", "Safety",
        "Comfort", "Design", "Connectivity", "Efficiency"
    ]

def get_theme_keywords(theme):
    """Get keywords associated with each theme"""
    keyword_map = {
        "Modern/Sleek": ["modern", "sleek", "contemporary", "clean", "streamlined", "stylish"],
        "Urban/City": ["urban", "city", "cityscape", "metropolitan", "downtown", "street"],
        "Performance/Sporty": ["performance", "sporty", "speed", "racing", "athletic", "powerful", "fast"],
        "Sophisticated": ["sophisticated", "elegant", "refined", "premium", "upscale", "classy"],
        "Innovative/Tech": ["innovative", "technology", "tech", "advanced", "cutting-edge", "smart"],
        "Dynamic": ["dynamic", "energetic", "vibrant", "active", "motion", "movement"],
        "Eco-Friendly": ["eco", "green", "sustainable", "environmental", "clean", "electric"],
        "Luxury": ["luxury", "luxurious", "premium", "high-end", "exclusive", "prestige"],
        "Contemporary": ["contemporary", "current", "today", "now", "present", "latest"],
        "Minimalist": ["minimalist", "simple", "clean", "uncluttered", "minimal", "pure"],
        "Bold/Striking": ["bold", "striking", "dramatic", "eye-catching", "powerful", "impressive"],
        "Futuristic": ["futuristic", "future", "tomorrow", "next-gen", "advanced", "revolutionary"],
        "Nature/Outdoor": ["nature", "outdoor", "landscape", "scenic", "natural", "countryside"],
        "Family-Oriented": ["family", "practical", "spacious", "comfortable", "safe", "reliable"],
        "Professional": ["professional", "business", "executive", "corporate", "work", "office"],
        "Safety": ["safety", "secure", "protection", "safe", "reliable", "trusted"],
        "Comfort": ["comfort", "comfortable", "cozy", "relaxing", "smooth", "pleasant"],
        "Design": ["design", "aesthetic", "beautiful", "attractive", "visual", "appearance"],
        "Connectivity": ["connected", "connectivity", "digital", "online", "network", "smart"],
        "Efficiency": ["efficient", "efficiency", "economical", "optimized", "smart", "intelligent"]
    }
    
    return keyword_map.get(theme, [theme.lower().replace("/", " ").split()])

def extract_and_categorize_themes(df, themes):
    """Extract themes from OpenAI summaries and categorize them"""
    results = []
    
    for idx, row in df.iterrows():
        summary = str(row['openai_summary']).lower()
        car_model = row.get('matched_car_models', 'Unknown')
        
        # Find themes in the summary
        found_themes = []
        theme_details = {}
        
        for theme in themes:
            keywords = get_theme_keywords(theme)
            found_keywords = []
            
            for keyword in keywords:
                if keyword in summary:
                    found_keywords.append(keyword)
            
            if found_keywords:
                found_themes.append(theme)
                theme_details[theme] = found_keywords
        
        results.append({
            'ad_id': row.get('ad_archive_id'),
            'car_model': car_model,
            'themes': found_themes,
            'theme_count': len(found_themes),
            'theme_details': theme_details,
            'summary_excerpt': summary[:200] + "..." if len(summary) > 200 else summary
        })
    
    return results

def analyze_theme_frequency(results):
    """Analyze theme frequency overall and by car model"""
    
    # Overall theme frequency
    all_themes = []
    for item in results:
        all_themes.extend(item['themes'])
    
    theme_counts = Counter(all_themes)
    
    # Theme frequency by car model
    model_themes = {}
    for item in results:
        model = item['car_model'] if pd.notna(item['car_model']) and item['car_model'] != 'Unknown' else 'Unknown'
        if model not in model_themes:
            model_themes[model] = []
        model_themes[model].extend(item['themes'])
    
    model_theme_counts = {}
    for model, themes in model_themes.items():
        if themes:  # Only include models with themes
            model_theme_counts[model] = Counter(themes)
    
    return theme_counts, model_theme_counts

def display_results(theme_counts, model_theme_counts):
    """Display the analysis results"""
    
    print("\n" + "="*60)
    print("OVERALL THEME FREQUENCY ANALYSIS")
    print("="*60)
    
    total_theme_mentions = sum(theme_counts.values())
    print(f"Total theme mentions across all ads: {total_theme_mentions}")
    print(f"Unique themes found: {len(theme_counts)}")
    
    print("\nTop Themes (with frequency):")
    for theme, count in theme_counts.most_common():
        percentage = (count / total_theme_mentions) * 100
        print(f"{theme:20} | {count:4d} mentions ({percentage:5.1f}%)")
    
    print("\n" + "="*60)
    print("THEME FREQUENCY BY CAR MODEL")
    print("="*60)
    
    # Sort models by total theme mentions
    model_totals = {model: sum(counts.values()) for model, counts in model_theme_counts.items()}
    sorted_models = sorted(model_totals.items(), key=lambda x: x[1], reverse=True)
    
    for model, total in sorted_models:
        if model != 'Unknown' and total > 0:
            print(f"\n{model} ({total} total theme mentions):")
            counts = model_theme_counts[model]
            for theme, count in counts.most_common(10):  # Top 10 themes per model
                percentage = (count / total) * 100
                print(f"  {theme:18} | {count:3d} ({percentage:5.1f}%)")

def save_detailed_results(results, theme_counts, model_theme_counts):
    """Save detailed results to CSV files"""
    
    # Save individual ad results
    results_df = pd.DataFrame(results)
    results_df.to_csv('detailed_theme_analysis.csv', index=False)
    
    # Save overall theme frequency
    theme_freq_df = pd.DataFrame([
        {'theme': theme, 'frequency': count, 'percentage': (count/sum(theme_counts.values()))*100}
        for theme, count in theme_counts.most_common()
    ])
    theme_freq_df.to_csv('theme_frequency_overall.csv', index=False)
    
    # Save model-specific theme frequency
    model_theme_data = []
    for model, counts in model_theme_counts.items():
        total = sum(counts.values())
        for theme, count in counts.items():
            model_theme_data.append({
                'car_model': model,
                'theme': theme,
                'frequency': count,
                'percentage': (count/total)*100
            })
    
    model_theme_df = pd.DataFrame(model_theme_data)
    model_theme_df.to_csv('theme_frequency_by_model.csv', index=False)
    
    print(f"\nDetailed results saved to:")
    print(f"- detailed_theme_analysis.csv")
    print(f"- theme_frequency_overall.csv") 
    print(f"- theme_frequency_by_model.csv")

def main():
    print("Car Advertisement Theme Analysis")
    print("="*40)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Get predefined themes
    themes = get_predefined_themes()
    print(f"\nAnalyzing for {len(themes)} predefined themes:")
    print(", ".join(themes))
    
    # Extract and categorize themes
    print(f"\nProcessing {len(df)} advertisements...")
    results = extract_and_categorize_themes(df, themes)
    
    # Analyze frequency
    theme_counts, model_theme_counts = analyze_theme_frequency(results)
    
    # Display results
    display_results(theme_counts, model_theme_counts)
    
    # Save results
    save_detailed_results(results, theme_counts, model_theme_counts)

if __name__ == "__main__":
    main()
