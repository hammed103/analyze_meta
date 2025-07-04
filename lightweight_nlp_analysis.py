import pandas as pd
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

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

def preprocess_text(text):
    """Clean and preprocess text for NLP analysis"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits, keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_custom_stop_words():
    """Get custom stop words for car advertisements"""
    # Basic English stop words
    basic_stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
        'this', 'these', 'they', 'them', 'their', 'there', 'where', 'when', 'what', 'who',
        'how', 'why', 'which', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
        'shall', 'will', 'do', 'does', 'did', 'have', 'had', 'having', 'been', 'being',
        'or', 'but', 'if', 'then', 'than', 'so', 'very', 'just', 'now', 'only', 'also',
        'all', 'any', 'some', 'no', 'not', 'more', 'most', 'other', 'such', 'own', 'same',
        'few', 'many', 'much', 'several', 'both', 'each', 'every', 'either', 'neither'
    }
    
    # Car-specific stop words
    car_stop_words = {
        'car', 'vehicle', 'electric', 'ev', 'auto', 'automotive', 'drive', 'driving',
        'advertisement', 'ad', 'brand', 'model', 'new', 'get', 'buy', 'purchase',
        'available', 'offer', 'price', 'cost', 'dealer', 'dealership', 'sale',
        'featuring', 'features', 'include', 'includes', 'comes', 'equipped',
        'page', 'facebook', 'instagram', 'social', 'media', 'post', 'content',
        'image', 'photo', 'picture', 'visual', 'shows', 'displays', 'depicts'
    }
    
    return basic_stop_words.union(car_stop_words)

def extract_themes_tfidf(texts, n_clusters=12, max_features=150):
    """Extract themes using TF-IDF and K-means clustering"""
    
    # Get stop words
    stop_words = list(get_custom_stop_words())
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=(1, 2),  # Include both single words and bigrams
        min_df=3,  # Ignore terms that appear in less than 3 documents
        max_df=0.7  # Ignore terms that appear in more than 70% of documents
    )
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top keywords by TF-IDF score
    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    keyword_scores = list(zip(feature_names, mean_scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    # Extract themes from clusters
    themes = []
    for i in range(n_clusters):
        # Get the centroid for this cluster
        centroid = kmeans.cluster_centers_[i]
        
        # Get top features for this cluster
        top_indices = centroid.argsort()[-8:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        top_scores = [centroid[idx] for idx in top_indices]
        
        # Create a theme name
        theme_name = create_theme_name(top_words[:3])
        
        themes.append({
            'theme_name': theme_name,
            'keywords': top_words,
            'scores': top_scores,
            'cluster_id': i
        })
    
    return themes, cluster_labels, keyword_scores

def create_theme_name(top_words):
    """Create a meaningful theme name from top words"""
    
    # Define theme mappings
    theme_patterns = {
        'performance': ['performance', 'power', 'speed', 'sporty', 'dynamic'],
        'design': ['design', 'style', 'aesthetic', 'sleek', 'modern'],
        'family': ['family', 'space', 'comfort', 'spacious', 'practical'],
        'technology': ['technology', 'tech', 'smart', 'digital', 'connectivity'],
        'luxury': ['luxury', 'premium', 'elegant', 'sophisticated', 'upscale'],
        'eco': ['eco', 'green', 'sustainable', 'environmental', 'clean'],
        'safety': ['safety', 'secure', 'protection', 'reliable', 'trusted'],
        'urban': ['urban', 'city', 'metropolitan', 'downtown', 'street'],
        'innovation': ['innovative', 'advanced', 'cutting', 'future', 'revolutionary'],
        'efficiency': ['efficient', 'optimized', 'economical', 'smart', 'intelligent']
    }
    
    # Check for theme matches
    top_words_str = ' '.join(top_words).lower()
    
    for theme, keywords in theme_patterns.items():
        if any(keyword in top_words_str for keyword in keywords):
            return theme.title()
    
    # If no clear match, use the most meaningful words
    meaningful_words = [word for word in top_words if len(word) > 3]
    if len(meaningful_words) >= 2:
        return f"{meaningful_words[0].title()}/{meaningful_words[1].title()}"
    elif meaningful_words:
        return meaningful_words[0].title()
    else:
        return f"Theme_{top_words[0].title()}"

def analyze_by_car_model(df, cluster_labels, themes):
    """Analyze theme distribution by car model"""
    
    # Add cluster labels to dataframe
    df_analysis = df.copy()
    df_analysis['theme_cluster'] = cluster_labels
    
    # Map cluster IDs to theme names
    cluster_to_theme = {theme['cluster_id']: theme['theme_name'] for theme in themes}
    df_analysis['theme_name'] = df_analysis['theme_cluster'].map(cluster_to_theme)
    
    # Analyze by car model
    model_theme_counts = {}
    
    for model in df_analysis['matched_car_models'].dropna().unique():
        if model and model != 'Unknown':
            model_data = df_analysis[df_analysis['matched_car_models'] == model]
            theme_counts = model_data['theme_name'].value_counts()
            if len(theme_counts) > 0:
                model_theme_counts[model] = theme_counts
    
    return model_theme_counts, df_analysis

def display_results(themes, keyword_scores, model_theme_counts):
    """Display the analysis results"""
    
    print("\n" + "="*70)
    print("LIGHTWEIGHT NLP THEME ANALYSIS RESULTS")
    print("="*70)
    
    print("\n1. TOP KEYWORDS (TF-IDF Analysis):")
    print("-" * 50)
    for keyword, score in keyword_scores[:25]:
        print(f"{keyword:30} | {score:.4f}")
    
    print("\n2. DISCOVERED THEMES:")
    print("-" * 50)
    for theme in themes:
        print(f"\n{theme['theme_name']}:")
        top_keywords = theme['keywords'][:5]
        print(f"  Keywords: {', '.join(top_keywords)}")
    
    print("\n3. THEME DISTRIBUTION BY CAR MODEL:")
    print("-" * 50)
    
    # Sort models by total ads
    model_totals = {model: counts.sum() for model, counts in model_theme_counts.items()}
    sorted_models = sorted(model_totals.items(), key=lambda x: x[1], reverse=True)
    
    for model, total in sorted_models[:15]:  # Top 15 models
        print(f"\n{model} ({total} ads):")
        theme_counts = model_theme_counts[model]
        for theme, count in theme_counts.head(5).items():
            percentage = (count / total) * 100
            print(f"  {theme:20} | {count:3d} ({percentage:5.1f}%)")

def save_results(themes, keyword_scores, model_theme_counts, df_analysis):
    """Save analysis results to CSV files"""
    
    # Save discovered themes
    themes_df = pd.DataFrame([
        {
            'theme_name': theme['theme_name'],
            'top_keywords': ', '.join(theme['keywords'][:5]),
            'cluster_id': theme['cluster_id']
        }
        for theme in themes
    ])
    themes_df.to_csv('lightweight_nlp_themes.csv', index=False)
    
    # Save top keywords
    keywords_df = pd.DataFrame(keyword_scores[:50], columns=['keyword', 'tfidf_score'])
    keywords_df.to_csv('lightweight_nlp_keywords.csv', index=False)
    
    # Save model-theme analysis
    model_theme_data = []
    for model, theme_counts in model_theme_counts.items():
        total = theme_counts.sum()
        for theme, count in theme_counts.items():
            model_theme_data.append({
                'car_model': model,
                'theme': theme,
                'frequency': count,
                'percentage': (count/total)*100
            })
    
    model_theme_df = pd.DataFrame(model_theme_data)
    model_theme_df.to_csv('lightweight_nlp_model_themes.csv', index=False)
    
    # Save detailed assignments
    df_analysis[['ad_archive_id', 'matched_car_models', 'theme_name', 'theme_cluster']].to_csv(
        'lightweight_nlp_detailed.csv', index=False
    )
    
    print(f"\nResults saved to:")
    print(f"- lightweight_nlp_themes.csv")
    print(f"- lightweight_nlp_keywords.csv") 
    print(f"- lightweight_nlp_model_themes.csv")
    print(f"- lightweight_nlp_detailed.csv")

def main():
    print("Lightweight NLP Theme Analysis for Car Advertisements")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess text data
    print("\nPreprocessing text data...")
    df['processed_summary'] = df['openai_summary'].apply(preprocess_text)
    
    # Remove empty summaries
    df_clean = df[df['processed_summary'].str.len() > 10].copy()
    print(f"Processing {len(df_clean)} ads with substantial text content")
    
    # Extract themes
    print("\nExtracting themes using TF-IDF and clustering...")
    themes, cluster_labels, keyword_scores = extract_themes_tfidf(
        df_clean['processed_summary'].tolist(), 
        n_clusters=12
    )
    
    # Analyze by car model
    print("Analyzing theme distribution by car model...")
    model_theme_counts, df_analysis = analyze_by_car_model(df_clean, cluster_labels, themes)
    
    # Display results
    display_results(themes, keyword_scores, model_theme_counts)
    
    # Save results
    save_results(themes, keyword_scores, model_theme_counts, df_analysis)

if __name__ == "__main__":
    main()
