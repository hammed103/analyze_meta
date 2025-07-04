import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings

warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


def load_data():
    """Load the car advertisement data"""
    try:
        df = pd.read_csv(
            "Data/facebook_ads_electric_vehicles_with_openai_summaries_cached.csv"
        )
        print(f"Loaded {len(df)} rows of data")

        # Filter for rows that have OpenAI summaries
        df_with_summaries = df[df["openai_summary"].notna()].copy()
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

    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_keywords_tfidf(texts, max_features=100, ngram_range=(1, 2)):
    """Extract keywords using TF-IDF"""

    # Custom stop words for car ads
    custom_stop_words = set(
        [
            "car",
            "vehicle",
            "electric",
            "ev",
            "auto",
            "automotive",
            "drive",
            "driving",
            "advertisement",
            "ad",
            "brand",
            "model",
            "new",
            "get",
            "buy",
            "purchase",
            "available",
            "offer",
            "price",
            "cost",
            "dealer",
            "dealership",
            "sale",
            "featuring",
            "features",
            "include",
            "includes",
            "comes",
            "equipped",
        ]
    )

    # Combine with standard English stop words
    stop_words = set(stopwords.words("english")).union(custom_stop_words)

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(stop_words),
        ngram_range=ngram_range,
        min_df=5,  # Ignore terms that appear in less than 5 documents
        max_df=0.8,  # Ignore terms that appear in more than 80% of documents
    )

    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Get average TF-IDF scores
    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

    # Create keyword-score pairs
    keyword_scores = list(zip(feature_names, mean_scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)

    return keyword_scores, vectorizer, tfidf_matrix


def perform_topic_modeling(tfidf_matrix, feature_names, n_topics=10):
    """Perform topic modeling using Latent Dirichlet Allocation"""

    # Initialize LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, max_iter=100
    )

    # Fit the model
    lda.fit(tfidf_matrix)

    # Extract topics
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        # Get top words for this topic
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_scores = [topic[i] for i in top_words_idx]

        topics.append({"topic_id": topic_idx, "words": top_words, "scores": top_scores})

    return topics, lda


def extract_semantic_themes(texts, n_clusters=15):
    """Extract themes using clustering of TF-IDF vectors"""

    # Get TF-IDF representation
    keyword_scores, vectorizer, tfidf_matrix = extract_keywords_tfidf(
        texts, max_features=200
    )

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # Get cluster centers and interpret them
    feature_names = vectorizer.get_feature_names_out()
    cluster_themes = []

    for i in range(n_clusters):
        # Get the centroid for this cluster
        centroid = kmeans.cluster_centers_[i]

        # Get top features for this cluster
        top_indices = centroid.argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        top_scores = [centroid[idx] for idx in top_indices]

        # Create a theme name based on top words
        theme_name = create_theme_name(top_words[:3])

        cluster_themes.append(
            {
                "theme_name": theme_name,
                "keywords": top_words,
                "scores": top_scores,
                "cluster_id": i,
            }
        )

    return cluster_themes, cluster_labels, keyword_scores


def create_theme_name(top_words):
    """Create a meaningful theme name from top words"""

    # Define theme mappings based on common patterns
    theme_mappings = {
        ("performance", "power", "speed"): "Performance",
        ("design", "style", "aesthetic"): "Design",
        ("family", "space", "comfort"): "Family-Oriented",
        ("technology", "tech", "smart"): "Technology",
        ("luxury", "premium", "elegant"): "Luxury",
        ("eco", "green", "sustainable"): "Eco-Friendly",
        ("modern", "contemporary", "sleek"): "Modern",
        ("safety", "secure", "protection"): "Safety",
        ("urban", "city", "metropolitan"): "Urban",
        ("innovative", "advanced", "cutting"): "Innovation",
        ("dynamic", "energetic", "vibrant"): "Dynamic",
        ("sophisticated", "refined", "classy"): "Sophisticated",
        ("connectivity", "connected", "digital"): "Connectivity",
        ("efficiency", "efficient", "optimized"): "Efficiency",
        ("bold", "striking", "dramatic"): "Bold",
        ("minimalist", "simple", "clean"): "Minimalist",
    }

    # Check for matches
    top_words_str = " ".join(top_words).lower()

    for keywords, theme in theme_mappings.items():
        if any(keyword in top_words_str for keyword in keywords):
            return theme

    # If no match, create name from top 2 words
    return f"{top_words[0].title()}/{top_words[1].title()}"


def analyze_themes_by_model(df, cluster_labels, cluster_themes):
    """Analyze theme distribution by car model"""

    # Add cluster labels to dataframe
    df_analysis = df.copy()
    df_analysis["theme_cluster"] = cluster_labels

    # Map cluster IDs to theme names
    cluster_to_theme = {
        theme["cluster_id"]: theme["theme_name"] for theme in cluster_themes
    }
    df_analysis["theme_name"] = df_analysis["theme_cluster"].map(cluster_to_theme)

    # Analyze by car model
    model_theme_analysis = {}

    for model in df_analysis["matched_car_models"].dropna().unique():
        if model != "Unknown":
            model_data = df_analysis[df_analysis["matched_car_models"] == model]
            theme_counts = model_data["theme_name"].value_counts()
            model_theme_analysis[model] = theme_counts

    return model_theme_analysis, df_analysis


def display_nlp_results(cluster_themes, keyword_scores, model_theme_analysis):
    """Display the NLP analysis results"""

    print("\n" + "=" * 60)
    print("NLP-BASED THEME ANALYSIS RESULTS")
    print("=" * 60)

    print("\n1. TOP KEYWORDS (TF-IDF Analysis):")
    print("-" * 40)
    for keyword, score in keyword_scores[:20]:
        print(f"{keyword:25} | {score:.4f}")

    print("\n2. DISCOVERED THEMES (Clustering Analysis):")
    print("-" * 40)
    for theme in cluster_themes:
        print(f"\n{theme['theme_name']}:")
        top_keywords = theme["keywords"][:5]
        print(f"  Keywords: {', '.join(top_keywords)}")

    print("\n3. THEME DISTRIBUTION BY CAR MODEL:")
    print("-" * 40)

    # Sort models by total ads
    model_totals = {
        model: counts.sum() for model, counts in model_theme_analysis.items()
    }
    sorted_models = sorted(model_totals.items(), key=lambda x: x[1], reverse=True)

    for model, total in sorted_models[:10]:  # Top 10 models
        print(f"\n{model} ({total} ads):")
        theme_counts = model_theme_analysis[model]
        for theme, count in theme_counts.head(5).items():
            percentage = (count / total) * 100
            print(f"  {theme:20} | {count:3d} ({percentage:5.1f}%)")


def save_nlp_results(cluster_themes, keyword_scores, model_theme_analysis, df_analysis):
    """Save NLP analysis results to CSV files"""

    # Save discovered themes
    themes_df = pd.DataFrame(
        [
            {
                "theme_name": theme["theme_name"],
                "top_keywords": ", ".join(theme["keywords"][:5]),
                "cluster_id": theme["cluster_id"],
            }
            for theme in cluster_themes
        ]
    )
    themes_df.to_csv("nlp_discovered_themes.csv", index=False)

    # Save top keywords
    keywords_df = pd.DataFrame(keyword_scores[:50], columns=["keyword", "tfidf_score"])
    keywords_df.to_csv("nlp_top_keywords.csv", index=False)

    # Save model-theme analysis
    model_theme_data = []
    for model, theme_counts in model_theme_analysis.items():
        total = theme_counts.sum()
        for theme, count in theme_counts.items():
            model_theme_data.append(
                {
                    "car_model": model,
                    "theme": theme,
                    "frequency": count,
                    "percentage": (count / total) * 100,
                }
            )

    model_theme_df = pd.DataFrame(model_theme_data)
    model_theme_df.to_csv("nlp_model_theme_analysis.csv", index=False)

    # Save detailed analysis with theme assignments
    df_analysis[
        ["ad_archive_id", "matched_car_models", "theme_name", "theme_cluster"]
    ].to_csv("nlp_detailed_theme_assignments.csv", index=False)

    print(f"\nNLP analysis results saved to:")
    print(f"- nlp_discovered_themes.csv")
    print(f"- nlp_top_keywords.csv")
    print(f"- nlp_model_theme_analysis.csv")
    print(f"- nlp_detailed_theme_assignments.csv")


def main():
    print("Advanced NLP Theme Analysis for Car Advertisements")
    print("=" * 55)

    # Load data
    df = load_data()
    if df is None:
        return

    # Preprocess text data
    print("\nPreprocessing text data...")
    df["processed_summary"] = df["openai_summary"].apply(preprocess_text)

    # Remove empty summaries
    df_clean = df[df["processed_summary"].str.len() > 10].copy()
    print(f"Processing {len(df_clean)} ads with substantial text content")

    # Extract themes using clustering
    print("\nExtracting themes using TF-IDF and clustering...")
    cluster_themes, cluster_labels, keyword_scores = extract_semantic_themes(
        df_clean["processed_summary"].tolist(), n_clusters=12
    )

    # Analyze themes by car model
    print("Analyzing theme distribution by car model...")
    model_theme_analysis, df_analysis = analyze_themes_by_model(
        df_clean, cluster_labels, cluster_themes
    )

    # Display results
    display_nlp_results(cluster_themes, keyword_scores, model_theme_analysis)

    # Save results
    save_nlp_results(cluster_themes, keyword_scores, model_theme_analysis, df_analysis)


if __name__ == "__main__":
    main()
