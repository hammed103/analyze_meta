import pandas as pd
import re

# Read the CSV file
df = pd.read_csv("facebook_ads_electric_vehicles_with_openai_summaries.csv")


def extract_themes(summary):
    """Extract all themes from OpenAI summary text"""
    if pd.isna(summary):
        return []

    # Pattern to match **Theme:** format
    pattern = r"\*\*([^*]+):\*\*"
    themes = re.findall(pattern, summary)

    # Clean up the themes (remove extra spaces)
    themes = [theme.strip() for theme in themes]

    # Remove "Key Message/Slogan" from themes
    themes = [theme for theme in themes if theme != "Key Message/Slogan"]

    return themes


def extract_specific_theme_content(summary, theme_name):
    """Extract content for a specific theme"""
    if pd.isna(summary):
        return ""

    # Pattern to match **Theme:** followed by content until next ** or end
    pattern = rf"\*\*{re.escape(theme_name)}:\*\*\s*(.*?)(?=\*\*|$)"
    match = re.search(pattern, summary, re.DOTALL)

    if match:
        content = match.group(1).strip()
        # Clean up extra newlines and spaces
        content = re.sub(r"\n+", " ", content)
        content = re.sub(r"\s+", " ", content)
        return content
    return ""


# Extract all themes as lists
df["all_themes"] = df["openai_summary"].apply(extract_themes)

# Extract specific theme content
df["brand_product_focus"] = df["openai_summary"].apply(
    lambda x: extract_specific_theme_content(x, "Brand & Product Focus")
)

df["exterior_design"] = df["openai_summary"].apply(
    lambda x: extract_specific_theme_content(x, "Exterior Design")
)

# You can add more specific themes as needed
df["key_message_slogan"] = df["openai_summary"].apply(
    lambda x: extract_specific_theme_content(x, "Key Message/Slogan")
)

df["performance"] = df["openai_summary"].apply(
    lambda x: extract_specific_theme_content(x, "Performance")
)

df["range_charging"] = df["openai_summary"].apply(
    lambda x: extract_specific_theme_content(x, "Range/Charging")
)

df["interior_comfort"] = df["openai_summary"].apply(
    lambda x: extract_specific_theme_content(x, "Interior/Comfort")
)

df["safety_assistance"] = df["openai_summary"].apply(
    lambda x: extract_specific_theme_content(x, "Safety/Assistance")
)

df["connectivity_digital"] = df["openai_summary"].apply(
    lambda x: extract_specific_theme_content(x, "Connectivity/Digital")
)

df["infotainment_audio"] = df["openai_summary"].apply(
    lambda x: extract_specific_theme_content(x, "Infotainment/Audio")
)

# Create a series of all themes for value_counts
all_themes_list = []
for themes_list in df["all_themes"].dropna():
    all_themes_list.extend(themes_list)

themes_series = pd.Series(all_themes_list)

print("=== THEME VALUE COUNTS (Overall) ===")
theme_counts = themes_series.value_counts()
print(theme_counts)

print("\n" + "=" * 60)
print("=== THEME VALUE COUNTS BY CAR MODEL ===")
print("=" * 60)

# Get records with both themes and car models
df_with_models = df[
    (df["all_themes"].notna()) & (df["matched_car_models"].notna())
].copy()

# Group by car model and analyze themes
for car_model in df_with_models["matched_car_models"].unique():
    model_df = df_with_models[df_with_models["matched_car_models"] == car_model]

    # Collect all themes for this car model
    model_themes_list = []
    for themes_list in model_df["all_themes"]:
        model_themes_list.extend(themes_list)

    if model_themes_list:  # Only show if there are themes
        model_themes_series = pd.Series(model_themes_list)
        model_theme_counts = model_themes_series.value_counts()

        print(f"\n--- {car_model} ---")
        print(f"Total ads: {len(model_df)}")
        print(model_theme_counts)

# Save the enhanced dataset
df.to_csv("facebook_ads_with_extracted_themes.csv", index=False)
print(f"\n\nEnhanced dataset saved as 'facebook_ads_with_extracted_themes.csv'")
print(
    f"New columns added: all_themes, brand_product_focus, exterior_design, key_message_slogan, performance, range_charging, interior_comfort, safety_assistance, connectivity_digital, infotainment_audio"
)
