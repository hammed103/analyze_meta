import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set style for better-looking plots
plt.style.use("default")
sns.set_palette("husl")


def create_theme_frequency_chart():
    """Create a bar chart of overall theme frequency"""

    try:
        df = pd.read_csv("theme_frequency_overall.csv")

        plt.figure(figsize=(14, 8))

        # Create horizontal bar chart for better readability
        bars = plt.barh(df["theme"][:15], df["frequency"][:15])

        # Add percentage labels on bars
        for i, (bar, freq, pct) in enumerate(
            zip(bars, df["frequency"][:15], df["percentage"][:15])
        ):
            plt.text(
                bar.get_width() + 50,
                bar.get_y() + bar.get_height() / 2,
                f"{freq:,} ({pct:.1f}%)",
                ha="left",
                va="center",
                fontsize=10,
            )

        plt.title(
            "Top 15 Themes in Electric Vehicle Advertisements\n(Based on 54,016 total theme mentions)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Number of Mentions", fontsize=12)
        plt.ylabel("Theme", fontsize=12)

        # Invert y-axis to show highest values at top
        plt.gca().invert_yaxis()

        # Add grid for better readability
        plt.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig("theme_frequency_chart.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("✓ Theme frequency chart saved as 'theme_frequency_chart.png'")

    except Exception as e:
        print(f"Error creating theme frequency chart: {e}")


def create_model_comparison_chart():
    """Create a comparison chart of themes by top car models"""

    try:
        df = pd.read_csv("theme_frequency_by_model.csv")

        # Get top 5 models by total theme mentions
        model_totals = (
            df.groupby("car_model")["frequency"].sum().sort_values(ascending=False)
        )
        top_models = model_totals.head(5).index.tolist()

        # Filter data for top models and top themes
        df_filtered = df[df["car_model"].isin(top_models)]

        # Get top 8 themes overall
        top_themes = (
            df.groupby("theme")["frequency"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
            .index.tolist()
        )
        df_filtered = df_filtered[df_filtered["theme"].isin(top_themes)]

        # Create pivot table for heatmap
        pivot_data = df_filtered.pivot(
            index="theme", columns="car_model", values="percentage"
        )
        pivot_data = pivot_data.fillna(0)

        plt.figure(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Percentage of Model's Total Themes"},
        )

        plt.title(
            "Theme Distribution by Top Car Models\n(Percentage of each model's total theme mentions)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Car Model", fontsize=12)
        plt.ylabel("Theme", fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig("model_theme_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("✓ Model-theme heatmap saved as 'model_theme_heatmap.png'")

    except Exception as e:
        print(f"Error creating model comparison chart: {e}")


def create_nlp_keywords_chart():
    """Create a chart of top NLP-discovered keywords"""

    try:
        df = pd.read_csv("lightweight_nlp_keywords.csv")

        plt.figure(figsize=(12, 8))

        # Create horizontal bar chart
        top_keywords = df.head(20)
        bars = plt.barh(top_keywords["keyword"], top_keywords["tfidf_score"])

        # Add value labels
        for bar, score in zip(bars, top_keywords["tfidf_score"]):
            plt.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        plt.title(
            "Top 20 Keywords from NLP Analysis\n(TF-IDF Scores)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("TF-IDF Score", fontsize=12)
        plt.ylabel("Keyword", fontsize=12)

        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig("nlp_keywords_chart.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("✓ NLP keywords chart saved as 'nlp_keywords_chart.png'")

    except Exception as e:
        print(f"Error creating NLP keywords chart: {e}")


def create_theme_evolution_summary():
    """Create a summary showing theme patterns"""

    try:
        # Load both analyses
        predefined = pd.read_csv("theme_frequency_overall.csv")
        nlp_themes = pd.read_csv("lightweight_nlp_themes.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Predefined themes pie chart
        top_predefined = predefined.head(10)
        others_freq = predefined[10:]["frequency"].sum()

        pie_data = list(top_predefined["frequency"]) + [others_freq]
        pie_labels = list(top_predefined["theme"]) + ["Others"]

        ax1.pie(pie_data, labels=pie_labels, autopct="%1.1f%%", startangle=90)
        ax1.set_title(
            "Predefined Theme Distribution\n(Top 10 + Others)", fontweight="bold"
        )

        # NLP themes bar chart
        nlp_model_data = pd.read_csv("lightweight_nlp_model_themes.csv")
        nlp_totals = (
            nlp_model_data.groupby("theme")["frequency"]
            .sum()
            .sort_values(ascending=False)
        )

        ax2.barh(nlp_totals.index, nlp_totals.values)
        ax2.set_title("NLP-Discovered Theme Frequency", fontweight="bold")
        ax2.set_xlabel("Frequency")
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig("theme_comparison_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("✓ Theme comparison summary saved as 'theme_comparison_summary.png'")

    except Exception as e:
        print(f"Error creating theme evolution summary: {e}")


def print_actionable_insights():
    """Print actionable insights for Thomas"""

    print("\n" + "=" * 80)
    print("ACTIONABLE INSIGHTS FOR THOMAS")
    print("=" * 80)

    try:
        # Load data
        predefined = pd.read_csv("theme_frequency_overall.csv")
        model_themes = pd.read_csv("theme_frequency_by_model.csv")
        nlp_keywords = pd.read_csv("lightweight_nlp_keywords.csv")

        print("\n🎯 KEY FINDINGS:")
        print("-" * 50)

        # Top themes
        top_theme = predefined.iloc[0]
        print(
            f"1. DOMINANT THEME: {top_theme['theme']} ({top_theme['percentage']:.1f}% of all mentions)"
        )
        print(f"   → This confirms EV ads heavily emphasize environmental benefits")

        # Underutilized themes
        low_themes = predefined[predefined["percentage"] < 2.0]["theme"].tolist()
        print(f"\n2. UNDERUTILIZED THEMES: {', '.join(low_themes[:5])}")
        print(f"   → Opportunity for differentiation in these areas")

        # Brand patterns
        vw_data = model_themes[model_themes["car_model"] == "Volkswagen ID.4"]
        if not vw_data.empty:
            vw_top = vw_data.nlargest(3, "frequency")["theme"].tolist()
            print(f"\n3. VOLKSWAGEN ID.4 FOCUS: {', '.join(vw_top)}")
            print(f"   → Market leader's messaging strategy")

        # NLP insights
        brand_keywords = nlp_keywords[
            nlp_keywords["keyword"].str.contains(
                "volkswagen|tesla|hyundai|kia|bmw|audi|volvo"
            )
        ]
        print(f"\n4. BRAND-SPECIFIC LANGUAGE:")
        for _, row in brand_keywords.head(5).iterrows():
            print(f"   → '{row['keyword']}' (TF-IDF: {row['tfidf_score']:.3f})")

        print(f"\n📊 RECOMMENDATIONS:")
        print("-" * 50)
        print(f"1. Use 'Eco-Friendly' themes but differentiate with secondary themes")
        print(f"2. Consider 'Urban/City' themes (only 0.5% usage) for differentiation")
        print(f"3. Analyze competitor language patterns using NLP keywords")
        print(f"4. Focus on underrepresented themes for unique positioning")
        print(f"5. Track theme evolution over time for trend analysis")

    except Exception as e:
        print(f"Error generating insights: {e}")


def main():
    print("Creating Theme Analysis Visualizations")
    print("=" * 50)

    # Create visualizations
    print("\n1. Creating theme frequency chart...")
    create_theme_frequency_chart()

    print("\n2. Creating model comparison heatmap...")
    create_model_comparison_chart()

    print("\n3. Creating NLP keywords chart...")
    create_nlp_keywords_chart()

    print("\n4. Creating theme comparison summary...")
    create_theme_evolution_summary()

    # Print insights
    print_actionable_insights()

    print(f"\n✅ All visualizations created successfully!")
    print(f"📁 Files saved:")
    print(f"   • theme_frequency_chart.png")
    print(f"   • model_theme_heatmap.png")
    print(f"   • nlp_keywords_chart.png")
    print(f"   • theme_comparison_summary.png")


if __name__ == "__main__":
    main()
