import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_all_results():
    """Load results from all theme analysis approaches"""
    results = {}
    
    try:
        # Load predefined theme analysis
        results['predefined'] = {
            'overall': pd.read_csv('theme_frequency_overall.csv'),
            'by_model': pd.read_csv('theme_frequency_by_model.csv')
        }
        print("✓ Loaded predefined theme analysis results")
    except:
        print("✗ Could not load predefined theme analysis results")
    
    try:
        # Load NLP-based analysis
        results['nlp'] = {
            'themes': pd.read_csv('lightweight_nlp_themes.csv'),
            'keywords': pd.read_csv('lightweight_nlp_keywords.csv'),
            'by_model': pd.read_csv('lightweight_nlp_model_themes.csv')
        }
        print("✓ Loaded NLP-based theme analysis results")
    except:
        print("✗ Could not load NLP-based theme analysis results")
    
    return results

def compare_approaches(results):
    """Compare the different theme analysis approaches"""
    
    print("\n" + "="*80)
    print("COMPARISON OF THEME ANALYSIS APPROACHES")
    print("="*80)
    
    if 'predefined' in results and 'nlp' in results:
        print("\n1. PREDEFINED THEMES vs NLP-DISCOVERED THEMES")
        print("-" * 60)
        
        # Top predefined themes
        predefined_top = results['predefined']['overall'].head(10)
        print("\nTop Predefined Themes:")
        for _, row in predefined_top.iterrows():
            print(f"  {row['theme']:20} | {row['frequency']:4d} ({row['percentage']:5.1f}%)")
        
        # NLP discovered themes
        nlp_themes = results['nlp']['themes']
        print(f"\nNLP-Discovered Themes ({len(nlp_themes)} total):")
        for _, row in nlp_themes.iterrows():
            print(f"  {row['theme_name']:20} | Keywords: {row['top_keywords']}")
    
    print("\n2. KEY INSIGHTS FROM EACH APPROACH")
    print("-" * 60)
    
    if 'predefined' in results:
        predefined_data = results['predefined']['overall']
        total_mentions = predefined_data['frequency'].sum()
        print(f"\nPredefined Theme Analysis:")
        print(f"  • Total theme mentions: {total_mentions:,}")
        print(f"  • Top theme: {predefined_data.iloc[0]['theme']} ({predefined_data.iloc[0]['percentage']:.1f}%)")
        print(f"  • Most diverse themes across all car models")
        print(f"  • Good for standardized comparison")
    
    if 'nlp' in results:
        nlp_keywords = results['nlp']['keywords']
        print(f"\nNLP-Based Analysis:")
        print(f"  • Top keyword: '{nlp_keywords.iloc[0]['keyword']}' (TF-IDF: {nlp_keywords.iloc[0]['tfidf_score']:.4f})")
        print(f"  • Discovered {len(results['nlp']['themes'])} distinct themes")
        print(f"  • Reveals actual language patterns in ads")
        print(f"  • Uncovers brand-specific messaging")

def analyze_brand_positioning(results):
    """Analyze how different brands position themselves"""
    
    print("\n3. BRAND POSITIONING INSIGHTS")
    print("-" * 60)
    
    if 'predefined' in results:
        model_data = results['predefined']['by_model']
        
        # Get top models by ad volume
        model_volumes = model_data.groupby('car_model')['frequency'].sum().sort_values(ascending=False)
        
        print("\nTop Car Models by Theme Mentions (Predefined Analysis):")
        for model, total in model_volumes.head(5).items():
            print(f"\n{model} ({total} total theme mentions):")
            model_themes = model_data[model_data['car_model'] == model].nlargest(3, 'frequency')
            for _, row in model_themes.iterrows():
                print(f"  {row['theme']:18} | {row['frequency']:3d} ({row['percentage']:5.1f}%)")
    
    if 'nlp' in results:
        nlp_model_data = results['nlp']['by_model']
        
        print(f"\nNLP-Discovered Brand Patterns:")
        
        # Analyze brand-specific themes
        brand_patterns = {}
        for _, row in nlp_model_data.iterrows():
            model = row['car_model']
            if model not in brand_patterns:
                brand_patterns[model] = []
            if row['percentage'] > 20:  # Only significant themes
                brand_patterns[model].append(f"{row['theme']} ({row['percentage']:.1f}%)")
        
        for model, patterns in list(brand_patterns.items())[:5]:
            if patterns:
                print(f"  {model}: {', '.join(patterns)}")

def extract_key_insights(results):
    """Extract key insights from the analysis"""
    
    print("\n4. KEY INSIGHTS & RECOMMENDATIONS")
    print("-" * 60)
    
    insights = []
    
    if 'predefined' in results:
        predefined_data = results['predefined']['overall']
        top_theme = predefined_data.iloc[0]
        insights.append(f"• {top_theme['theme']} is the dominant theme across all EV ads ({top_theme['percentage']:.1f}%)")
        
        # Find themes with low representation
        low_themes = predefined_data[predefined_data['percentage'] < 2.0]
        if len(low_themes) > 0:
            insights.append(f"• Underutilized themes: {', '.join(low_themes['theme'].tolist())}")
    
    if 'nlp' in results:
        nlp_keywords = results['nlp']['keywords']
        top_keywords = nlp_keywords.head(5)['keyword'].tolist()
        insights.append(f"• Most important language patterns: {', '.join(top_keywords)}")
        
        # Brand concentration
        nlp_model_data = results['nlp']['by_model']
        brand_concentration = nlp_model_data.groupby('car_model')['frequency'].sum().sort_values(ascending=False)
        top_brand = brand_concentration.index[0]
        insights.append(f"• {top_brand} dominates the advertising volume")
    
    print("\nKey Findings:")
    for insight in insights:
        print(f"  {insight}")
    
    print(f"\nRecommendations for Thomas:")
    print(f"  • Use predefined themes for standardized frequency analysis")
    print(f"  • Use NLP themes to understand actual messaging patterns")
    print(f"  • Focus on underrepresented themes for differentiation opportunities")
    print(f"  • Analyze competitor messaging using NLP-discovered patterns")

def create_theme_frequency_summary():
    """Create a summary of theme frequencies for easy reference"""
    
    try:
        # Load predefined results
        overall_themes = pd.read_csv('theme_frequency_overall.csv')
        
        print(f"\n5. THEME FREQUENCY SUMMARY (Value Counts Format)")
        print("-" * 60)
        print(f"Based on {overall_themes['frequency'].sum():,} total theme mentions:")
        
        for _, row in overall_themes.iterrows():
            print(f"{row['theme']:20} {row['frequency']:5d}")
            
    except Exception as e:
        print(f"Could not create summary: {e}")

def main():
    print("Theme Analysis Summary & Comparison")
    print("="*50)
    
    # Load all results
    results = load_all_results()
    
    if not results:
        print("No analysis results found. Please run the theme analysis scripts first.")
        return
    
    # Compare approaches
    compare_approaches(results)
    
    # Analyze brand positioning
    analyze_brand_positioning(results)
    
    # Extract insights
    extract_key_insights(results)
    
    # Create frequency summary
    create_theme_frequency_summary()

if __name__ == "__main__":
    main()
