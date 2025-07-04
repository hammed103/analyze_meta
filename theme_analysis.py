import pandas as pd
import numpy as np
from collections import Counter
import re
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_data():
    """Load the car advertisement data"""
    try:
        df = pd.read_csv('Data/facebook_ads_electric_vehicles_with_openai_summaries_cached.csv')
        print(f"Loaded {len(df)} rows of data")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def extract_visual_descriptions(df):
    """Extract visual descriptions from OpenAI summaries"""
    visual_descriptions = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('openai_summary')):
            summary = str(row['openai_summary'])
            
            # Look for visual description patterns
            # Common patterns: "features a...", "shows...", "displays...", "visual style..."
            visual_patterns = [
                r'features a ([^.]+)',
                r'shows ([^.]+)',
                r'displays ([^.]+)', 
                r'visual style is ([^.]+)',
                r'emphasizing ([^.]+)',
                r'focusing on ([^.]+)',
                r'appeal to ([^.]+)'
            ]
            
            description = ""
            for pattern in visual_patterns:
                matches = re.findall(pattern, summary, re.IGNORECASE)
                if matches:
                    description += " ".join(matches) + " "
            
            if description.strip():
                visual_descriptions.append({
                    'ad_id': row.get('ad_archive_id'),
                    'car_model': row.get('matched_car_models'),
                    'visual_description': description.strip(),
                    'full_summary': summary
                })
    
    return visual_descriptions

def analyze_themes_with_openai(visual_descriptions, sample_size=50):
    """Use OpenAI to identify common themes from visual descriptions"""
    
    # Take a sample for theme identification
    sample_descriptions = visual_descriptions[:sample_size] if len(visual_descriptions) > sample_size else visual_descriptions
    
    # Combine descriptions for analysis
    combined_text = "\n".join([desc['visual_description'] for desc in sample_descriptions])
    
    prompt = f"""
    Analyze the following car advertisement visual descriptions and identify the most common themes. 
    Return ONLY a Python list of discrete theme categories (10-15 themes max) that capture the main visual and conceptual elements.
    
    Focus on themes like:
    - Design aesthetics (modern, sleek, futuristic, etc.)
    - Environment/setting (urban, nature, highway, etc.) 
    - Emotions/appeals (sophistication, innovation, performance, etc.)
    - Visual style (dynamic, minimalist, bold, etc.)
    
    Descriptions to analyze:
    {combined_text}
    
    Return format: ["Theme1", "Theme2", "Theme3", ...]
    """
    
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # Extract the list from the response
        theme_list_str = response.choices[0].message.content.strip()
        # Use eval carefully - in production, use ast.literal_eval
        themes = eval(theme_list_str)
        return themes
        
    except Exception as e:
        print(f"Error with OpenAI analysis: {e}")
        # Fallback manual themes
        return [
            "Modern/Sleek", "Urban/City", "Performance/Sporty", "Sophisticated", 
            "Innovative/Tech", "Dynamic", "Eco-Friendly", "Luxury", 
            "Contemporary", "Minimalist", "Bold/Striking", "Futuristic",
            "Nature/Outdoor", "Family-Oriented", "Professional"
        ]

def categorize_descriptions(visual_descriptions, themes):
    """Categorize each visual description by themes"""
    categorized_data = []
    
    for desc in visual_descriptions:
        description_text = desc['visual_description'].lower()
        found_themes = []
        
        for theme in themes:
            # Create keywords for each theme
            theme_keywords = get_theme_keywords(theme)
            
            for keyword in theme_keywords:
                if keyword.lower() in description_text:
                    found_themes.append(theme)
                    break
        
        categorized_data.append({
            'ad_id': desc['ad_id'],
            'car_model': desc['car_model'],
            'visual_description': desc['visual_description'],
            'themes': found_themes,
            'theme_count': len(found_themes)
        })
    
    return categorized_data

def get_theme_keywords(theme):
    """Get keywords associated with each theme"""
    keyword_map = {
        "Modern/Sleek": ["modern", "sleek", "contemporary", "clean", "streamlined"],
        "Urban/City": ["urban", "city", "cityscape", "metropolitan", "downtown"],
        "Performance/Sporty": ["performance", "sporty", "speed", "racing", "athletic"],
        "Sophisticated": ["sophisticated", "elegant", "refined", "premium", "upscale"],
        "Innovative/Tech": ["innovative", "technology", "tech", "advanced", "cutting-edge"],
        "Dynamic": ["dynamic", "energetic", "vibrant", "active", "motion"],
        "Eco-Friendly": ["eco", "green", "sustainable", "environmental", "clean energy"],
        "Luxury": ["luxury", "luxurious", "premium", "high-end", "exclusive"],
        "Contemporary": ["contemporary", "current", "today", "now", "present"],
        "Minimalist": ["minimalist", "simple", "clean", "uncluttered", "minimal"],
        "Bold/Striking": ["bold", "striking", "dramatic", "eye-catching", "powerful"],
        "Futuristic": ["futuristic", "future", "tomorrow", "next-gen", "advanced"],
        "Nature/Outdoor": ["nature", "outdoor", "landscape", "scenic", "natural"],
        "Family-Oriented": ["family", "practical", "spacious", "comfortable", "safe"],
        "Professional": ["professional", "business", "executive", "corporate", "work"]
    }
    
    return keyword_map.get(theme, [theme.lower().replace("/", " ").split()])

def analyze_theme_frequency(categorized_data):
    """Analyze theme frequency overall and by car model"""
    
    # Overall theme frequency
    all_themes = []
    for item in categorized_data:
        all_themes.extend(item['themes'])
    
    theme_counts = Counter(all_themes)
    
    # Theme frequency by car model
    model_themes = {}
    for item in categorized_data:
        model = item['car_model'] if pd.notna(item['car_model']) else 'Unknown'
        if model not in model_themes:
            model_themes[model] = []
        model_themes[model].extend(item['themes'])
    
    model_theme_counts = {}
    for model, themes in model_themes.items():
        model_theme_counts[model] = Counter(themes)
    
    return theme_counts, model_theme_counts

def main():
    print("Loading car advertisement data...")
    df = load_data()
    
    if df is None:
        return
    
    print("Extracting visual descriptions...")
    visual_descriptions = extract_visual_descriptions(df)
    print(f"Found {len(visual_descriptions)} ads with visual descriptions")
    
    if len(visual_descriptions) == 0:
        print("No visual descriptions found. Check the data format.")
        return
    
    print("Analyzing themes with OpenAI...")
    themes = analyze_themes_with_openai(visual_descriptions)
    print(f"Identified themes: {themes}")
    
    print("Categorizing descriptions by themes...")
    categorized_data = categorize_descriptions(visual_descriptions, themes)
    
    print("Analyzing theme frequency...")
    theme_counts, model_theme_counts = analyze_theme_frequency(categorized_data)
    
    # Display results
    print("\n" + "="*50)
    print("OVERALL THEME FREQUENCY")
    print("="*50)
    for theme, count in theme_counts.most_common():
        print(f"{theme}: {count}")
    
    print("\n" + "="*50)
    print("THEME FREQUENCY BY CAR MODEL")
    print("="*50)
    for model, counts in model_theme_counts.items():
        if model != 'Unknown' and sum(counts.values()) > 0:
            print(f"\n{model}:")
            for theme, count in counts.most_common(5):  # Top 5 themes per model
                print(f"  {theme}: {count}")
    
    # Save results
    results_df = pd.DataFrame(categorized_data)
    results_df.to_csv('theme_analysis_results.csv', index=False)
    print(f"\nResults saved to theme_analysis_results.csv")

if __name__ == "__main__":
    main()
