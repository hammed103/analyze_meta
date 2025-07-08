#!/usr/bin/env python3
"""
Electric Vehicle Ads Analysis Dashboard
Interactive Streamlit dashboard for analyzing Facebook EV ads data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from PIL import Image
import io
import os
import glob
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Meta Analysis - EV Ads Dashboard",
    page_icon="🚗⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_data_from_chunks():
    """Load data from chunked files in the Data directory."""
    import glob

    data_dir = "Data"
    chunk_pattern = os.path.join(data_dir, "facebook_ads_chunk_*.csv")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        return None

    st.info(f"📁 Loading data from {len(chunk_files)} chunk files in {data_dir}/")

    # Load and combine all chunks
    df_list = []
    progress_bar = st.progress(0)

    for i, file in enumerate(chunk_files):
        df_chunk = pd.read_csv(file)
        df_list.append(df_chunk)
        progress_bar.progress((i + 1) / len(chunk_files))

    combined_df = pd.concat(df_list, ignore_index=True)
    st.success(
        f"✅ Successfully combined {len(chunk_files)} chunks into {len(combined_df)} total rows"
    )

    return combined_df


def check_for_large_file_or_chunks():
    """Check for the large file or chunked data and provide guidance."""
    filename = "facebook_ads_electric_vehicles_with_openai_summaries_cached.csv"
    data_dir = "Data"

    # Check if large file exists
    if os.path.exists(filename):
        return "large_file"

    # Check if chunks exist
    chunk_pattern = os.path.join(data_dir, "facebook_ads_chunk_*.csv")
    chunk_files = glob.glob(chunk_pattern)

    if chunk_files:
        return "chunks"

    # Neither exists - show instructions
    st.warning("📥 Large dataset not found.")
    st.info(
        """
    **To enable the full dataset:**

    **Option 1: Split the large file (Recommended)**
    1. Place `facebook_ads_electric_vehicles_with_openai_summaries_cached.csv` in this directory
    2. Run: `python split_large_file.py`
    3. This will create chunks in the `Data/` folder
    4. Refresh the app

    **Option 2: Use the large file directly**
    - Place `facebook_ads_electric_vehicles_with_openai_summaries_cached.csv` in this directory

    The app will continue with available smaller datasets.
    """
    )
    return "none"


@st.cache_data
def load_data():
    """Load and cache the EV ads data."""
    # Check what data sources are available
    data_source = check_for_large_file_or_chunks()

    try:
        # Try to load from chunks first, then large file, then fallback options
        if data_source == "chunks":
            df = load_data_from_chunks()
            if df is not None:
                return df

        elif data_source == "large_file":
            df = pd.read_csv(
                "facebook_ads_electric_vehicles_with_openai_summaries_cached.csv"
            )
            st.info("✓ Loaded full dataset with OpenAI summaries and image themes")
            return df

        # Fallback to other available datasets
        try:
            df = pd.read_csv(
                "facebook_ads_electric_vehicles_with_openai_summaries_cached.csv"
            )
            st.info("✓ Loaded dataset with OpenAI summaries and image themes")
        except FileNotFoundError:
            try:
                df = pd.read_csv(
                    "facebook_ads_electric_vehicles_with_openai_summaries_cached.csv"
                )
                st.info(
                    "✓ Loaded cleaned AI enhanced data with OpenAI summaries and image themes (optimized file size)"
                )
            except FileNotFoundError:
                try:
                    df = pd.read_csv(
                        "facebook_ads_electric_vehicles_with_openai_summaries_cached.csv"
                    )
                    st.info(
                        "✓ Loaded ultimate AI enhanced data with OpenAI summaries and image themes"
                    )
                except FileNotFoundError:
                    try:
                        df = pd.read_csv(
                            "facebook_ads_electric_vehicles_with_openai_summaries_cached.csv"
                        )
                        st.info("✓ Loaded data with OpenAI summaries")
                    except FileNotFoundError:
                        df = pd.read_csv("facebook_ads_electric_vehicles.csv")
                        # Add missing columns with default values
                        df["page_classification"] = "unknown"
                        st.info("✓ Loaded base EV ads data (no classifications)")

        # Ensure required columns exist
        required_columns = {
            "male_percentage": 0,
            "female_percentage": 0,
            "total_male_audience": 0,
            "total_female_audience": 0,
            "page_like_count": 0,
            "spend": None,
            "new_image_url": None,
            "ad_title": "",
            "ad_text": "",
            "cta_text": "",
            "display_format": "",
            "start_date": "",
            "end_date": "",
            "targeted_countries": "",
            "matched_car_models": "Unknown",
            "openai_summary": "",
            "ad_theme": "",
        }

        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val

        return df
    except FileNotFoundError:
        st.error(
            "Data file not found. Please ensure 'facebook_ads_electric_vehicles.csv' is in the current directory."
        )
        return None


def load_image_from_url(url, timeout=5):
    """Load image from URL with error handling."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        return None


def find_local_image(ad_id, image_url):
    """Find locally saved image for an ad"""
    import hashlib

    # Generate the same filename used in download scripts
    url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
    filename = f"{ad_id}_{url_hash}.jpg"

    # Check possible local directories
    local_dirs = [
        "ev_ad_images/by_car_model",
        "ev_ad_images/thumbnails",
        "sample_images",
        "downloaded_images/originals",
    ]

    for base_dir in local_dirs:
        if os.path.exists(base_dir):
            # Search in subdirectories
            for root, dirs, files in os.walk(base_dir):
                if filename in files:
                    return os.path.join(root, filename)

            # Also check direct filename match
            direct_path = os.path.join(base_dir, filename)
            if os.path.exists(direct_path):
                return direct_path

    return None


def load_image_local_or_url(ad_id, image_url, prefer_local=True):
    """Load image from local file first, fallback to URL"""

    if prefer_local and pd.notna(image_url):
        # Try to find local image first
        local_path = find_local_image(ad_id, image_url)
        if local_path:
            try:
                image = Image.open(local_path)
                return image, "local"
            except Exception as e:
                pass  # Fall back to URL

    # Fallback to URL loading
    if pd.notna(image_url):
        image = load_image_from_url(image_url)
        if image:
            return image, "url"

    return None, "failed"


def analyze_car_model(df, model):
    """Analyze ads for a specific car model."""
    model_data = df[df["matched_car_models"] == model].copy()

    if len(model_data) == 0:
        return None

    # Safe column access with defaults
    def safe_column_analysis(column, default_value=None):
        if column in model_data.columns:
            return model_data[column]
        else:
            return pd.Series([default_value] * len(model_data))

    analysis = {
        "total_ads": len(model_data),
        "unique_advertisers": (
            model_data["page_name"].nunique()
            if "page_name" in model_data.columns
            else 0
        ),
        "top_advertisers": safe_column_analysis("page_name", "Unknown")
        .value_counts()
        .head(5),
        "cta_types": safe_column_analysis("cta_text", "Unknown")
        .value_counts()
        .head(10),
        "display_formats": safe_column_analysis(
            "display_format", "Unknown"
        ).value_counts(),
        "avg_male_targeting": safe_column_analysis("male_percentage", 0).mean(),
        "avg_female_targeting": safe_column_analysis("female_percentage", 0).mean(),
        "spend_data": safe_column_analysis("spend").dropna(),
        "date_range": {
            "start": safe_column_analysis("start_date", "").min(),
            "end": safe_column_analysis("end_date", "").max(),
        },
        "page_classifications": safe_column_analysis(
            "page_classification", "unknown"
        ).value_counts(),
        "countries": safe_column_analysis("targeted_countries", "")
        .astype(str)
        .str.split(";")
        .explode()
        .value_counts()
        .head(10),
    }

    return analysis, model_data


def create_cta_analysis_chart(df):
    """Create CTA analysis visualization."""
    cta_data = df["cta_text"].value_counts().head(15)

    fig = px.bar(
        x=cta_data.values,
        y=cta_data.index,
        orientation="h",
        title="Top Call-to-Action Texts",
        labels={"x": "Number of Ads", "y": "CTA Text"},
    )
    fig.update_layout(height=500)
    return fig


def create_gender_targeting_chart(df):
    """Create gender targeting analysis."""
    gender_data = df[df["male_percentage"] > 0].copy()

    fig = px.scatter(
        gender_data,
        x="male_percentage",
        y="female_percentage",
        color="matched_car_models",
        size="total_male_audience",
        hover_data=["page_name", "ad_title"],
        title="Gender Targeting by Car Model",
        labels={"male_percentage": "Male %", "female_percentage": "Female %"},
    )
    return fig


def create_advertiser_type_chart(df):
    """Create advertiser type distribution chart."""
    page_class_data = df["page_classification"].value_counts()

    fig = px.pie(
        values=page_class_data.values,
        names=page_class_data.index,
        title="Distribution of Advertiser Types",
    )
    return fig


# Theme Analysis Functions
@st.cache_data
def load_theme_analysis_data():
    """Load pre-computed theme analysis results (no recomputation)"""
    theme_data = {}
    files_found = []

    try:
        # Load predefined theme analysis
        if os.path.exists("theme_frequency_overall.csv"):
            theme_data["overall"] = pd.read_csv("theme_frequency_overall.csv")
            files_found.append("theme_frequency_overall.csv")
        if os.path.exists("theme_frequency_by_model.csv"):
            theme_data["by_model"] = pd.read_csv("theme_frequency_by_model.csv")
            files_found.append("theme_frequency_by_model.csv")

        # Load NLP analysis
        if os.path.exists("lightweight_nlp_keywords.csv"):
            theme_data["nlp_keywords"] = pd.read_csv("lightweight_nlp_keywords.csv")
            files_found.append("lightweight_nlp_keywords.csv")
        if os.path.exists("lightweight_nlp_model_themes.csv"):
            theme_data["nlp_by_model"] = pd.read_csv("lightweight_nlp_model_themes.csv")
            files_found.append("lightweight_nlp_model_themes.csv")
        if os.path.exists("lightweight_nlp_themes.csv"):
            theme_data["nlp_themes"] = pd.read_csv("lightweight_nlp_themes.csv")
            files_found.append("lightweight_nlp_themes.csv")

        # Store info about loaded files for display
        theme_data["_files_loaded"] = files_found

    except Exception as e:
        st.error(f"Error loading theme analysis data: {e}")

    return theme_data


def get_predefined_themes():
    """Return predefined themes for analysis"""
    return [
        "Eco-Friendly",
        "Design",
        "Performance/Sporty",
        "Family-Oriented",
        "Modern/Sleek",
        "Connectivity",
        "Safety",
        "Dynamic",
        "Comfort",
        "Innovative/Tech",
        "Contemporary",
        "Futuristic",
        "Minimalist",
        "Bold/Striking",
        "Nature/Outdoor",
        "Sophisticated",
        "Luxury",
        "Efficiency",
        "Professional",
        "Urban/City",
    ]


def display_theme_files_info(theme_data):
    """Display information about loaded theme analysis files"""
    if "_files_loaded" in theme_data and theme_data["_files_loaded"]:
        st.success(
            f"✅ Loaded {len(theme_data['_files_loaded'])} theme analysis files:"
        )
        for file in theme_data["_files_loaded"]:
            st.write(f"  • {file}")
    else:
        st.warning("⚠️ No pre-computed theme analysis files found.")
        st.info(
            """
        **To generate theme analysis data, run:**
        ```bash
        python3 simple_theme_analysis.py
        python3 lightweight_nlp_analysis.py
        ```
        """
        )
    return len(theme_data.get("_files_loaded", []))


def get_theme_keywords(theme):
    """Get keywords for each theme"""
    keyword_map = {
        "Eco-Friendly": [
            "eco",
            "green",
            "sustainable",
            "environmental",
            "clean",
            "electric",
        ],
        "Design": ["design", "aesthetic", "beautiful", "attractive", "visual", "style"],
        "Performance/Sporty": [
            "performance",
            "sporty",
            "speed",
            "racing",
            "athletic",
            "powerful",
        ],
        "Family-Oriented": [
            "family",
            "practical",
            "spacious",
            "comfortable",
            "safe",
            "reliable",
        ],
        "Modern/Sleek": [
            "modern",
            "sleek",
            "contemporary",
            "clean",
            "streamlined",
            "stylish",
        ],
        "Connectivity": [
            "connected",
            "connectivity",
            "digital",
            "online",
            "network",
            "smart",
        ],
        "Safety": ["safety", "secure", "protection", "safe", "reliable", "trusted"],
        "Dynamic": ["dynamic", "energetic", "vibrant", "active", "motion", "movement"],
        "Comfort": ["comfort", "comfortable", "cozy", "relaxing", "smooth", "pleasant"],
        "Innovative/Tech": [
            "innovative",
            "technology",
            "tech",
            "advanced",
            "cutting-edge",
            "smart",
        ],
        "Contemporary": [
            "contemporary",
            "current",
            "today",
            "now",
            "present",
            "latest",
        ],
        "Futuristic": [
            "futuristic",
            "future",
            "tomorrow",
            "next-gen",
            "advanced",
            "revolutionary",
        ],
        "Minimalist": [
            "minimalist",
            "simple",
            "clean",
            "uncluttered",
            "minimal",
            "pure",
        ],
        "Bold/Striking": [
            "bold",
            "striking",
            "dramatic",
            "eye-catching",
            "powerful",
            "impressive",
        ],
        "Nature/Outdoor": [
            "nature",
            "outdoor",
            "landscape",
            "scenic",
            "natural",
            "countryside",
        ],
        "Sophisticated": [
            "sophisticated",
            "elegant",
            "refined",
            "premium",
            "upscale",
            "classy",
        ],
        "Luxury": [
            "luxury",
            "luxurious",
            "premium",
            "high-end",
            "exclusive",
            "prestige",
        ],
        "Efficiency": [
            "efficient",
            "efficiency",
            "economical",
            "optimized",
            "smart",
            "intelligent",
        ],
        "Professional": [
            "professional",
            "business",
            "executive",
            "corporate",
            "work",
            "office",
        ],
        "Urban/City": [
            "urban",
            "city",
            "cityscape",
            "metropolitan",
            "downtown",
            "street",
        ],
    }
    return keyword_map.get(theme, [theme.lower().replace("/", " ").split()])


def main():
    st.title("🚗⚡ Meta Analysis - Electric Vehicle Ads Dashboard")
    st.markdown("---")

    # Load data
    df = load_data()
    if df is None:
        return

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Car model filter - handle mixed data types
    car_models = df["matched_car_models"].fillna("Unknown").astype(str).unique()
    car_models = sorted([model for model in car_models if model])
    selected_models = st.sidebar.multiselect(
        "Select Car Models",
        car_models,
        default=car_models,  # Select ALL car models by default
    )

    # Advertiser type filter - handle mixed data types
    advertiser_types = df["page_classification"].fillna("unknown").astype(str).unique()
    advertiser_types = sorted([atype for atype in advertiser_types if atype])
    selected_advertiser_types = st.sidebar.multiselect(
        "Select Advertiser Types",
        advertiser_types,
        default=advertiser_types,  # Select ALL advertiser types by default
    )

    # Filter data
    filtered_df = df[
        (df["matched_car_models"].isin(selected_models))
        & (df["page_classification"].isin(selected_advertiser_types))
    ]

    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Overview",
            "🚗 Car Model Analysis",
            "🎯 Ad Creative Analysis",
            "🖼️ Image Gallery",
            "🎨 Theme Analysis",
            "📈 Advanced Analytics",
        ]
    )

    with tab1:
        st.header("📊 Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Ads", len(filtered_df))

        with col2:
            st.metric("Car Models", filtered_df["matched_car_models"].nunique())

        with col3:
            st.metric("Advertisers", filtered_df["page_name"].nunique())

        with col4:
            avg_male_targeting = filtered_df["male_percentage"].mean()
            st.metric("Avg Male Targeting", f"{avg_male_targeting:.1f}%")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Top car models
            model_counts = filtered_df["matched_car_models"].value_counts().head(10)
            fig = px.bar(
                x=model_counts.values,
                y=model_counts.index,
                orientation="h",
                title="Top Electric Vehicle Models by Ad Count",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Advertiser type distribution
            fig = create_advertiser_type_chart(filtered_df)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("🚗 Car Model Deep Dive")

        # Model selector
        selected_model = st.selectbox(
            "Choose a car model for detailed analysis:", car_models
        )

        if selected_model:
            analysis, model_data = analyze_car_model(df, selected_model)

            if analysis:
                # Key metrics for the model
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Ads", analysis["total_ads"])

                with col2:
                    st.metric("Advertisers", analysis["unique_advertisers"])

                with col3:
                    st.metric(
                        "Male Targeting", f"{analysis['avg_male_targeting']:.1f}%"
                    )

                with col4:
                    st.metric(
                        "Female Targeting", f"{analysis['avg_female_targeting']:.1f}%"
                    )

                # Detailed analysis
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Top Advertisers")
                    st.bar_chart(analysis["top_advertisers"])

                    st.subheader("Display Formats")
                    st.bar_chart(analysis["display_formats"])

                with col2:
                    st.subheader("Call-to-Action Types")
                    cta_clean = analysis["cta_types"].dropna()
                    if len(cta_clean) > 0:
                        st.bar_chart(cta_clean)

                    st.subheader("Advertiser Types")
                    st.bar_chart(analysis["page_classifications"])

                # Sample ads table
                st.subheader("Sample Ads")
                sample_cols = [
                    "page_name",
                    "ad_title",
                    "ad_text",
                    "cta_text",
                    "start_date",
                    "page_classification",
                ]
                available_cols = [
                    col for col in sample_cols if col in model_data.columns
                ]
                st.dataframe(
                    model_data[available_cols].head(10), use_container_width=True
                )

    with tab3:
        st.header("🎯 Ad Creative Analysis")

        # CTA Analysis
        st.subheader("Call-to-Action Analysis")
        fig = create_cta_analysis_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True)

        # Gender targeting analysis
        st.subheader("Gender Targeting Patterns")
        fig = create_gender_targeting_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True)

        # Ad text analysis
        st.subheader("Ad Text Themes")

        # Simple keyword analysis
        all_text = " ".join(filtered_df["ad_text"].dropna().astype(str))
        words = all_text.lower().split()

        # Filter out common words and focus on car-related terms
        car_keywords = [
            word
            for word in words
            if len(word) > 3
            and any(
                term in word
                for term in [
                    "electric",
                    "ev",
                    "battery",
                    "charge",
                    "eco",
                    "green",
                    "hybrid",
                    "tesla",
                    "bmw",
                    "audi",
                    "volvo",
                    "volkswagen",
                ]
            )
        ]

        if car_keywords:
            keyword_counts = Counter(car_keywords).most_common(20)
            if keyword_counts:
                keywords_df = pd.DataFrame(keyword_counts, columns=["Keyword", "Count"])
                fig = px.bar(
                    keywords_df,
                    x="Count",
                    y="Keyword",
                    orientation="h",
                    title="Most Common Keywords in Ad Text",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("🖼️ Ad Image Gallery")

        # Get all images from filtered data
        all_image_data = filtered_df[filtered_df["new_image_url"].notna()].copy()
        total_images = len(all_image_data)

        if total_images == 0:
            st.info("No images found in the filtered data.")
        else:
            # Check how many images are available locally
            local_count = 0
            for _, row in all_image_data.iterrows():
                if find_local_image(row["ad_archive_id"], row["new_image_url"]):
                    local_count += 1

            # Show local vs remote status
            col_status1, col_status2 = st.columns(2)
            with col_status1:
                st.metric("💾 Local Images", f"{local_count}/{total_images}")
            with col_status2:
                st.metric(
                    "🌐 Remote Images", f"{total_images - local_count}/{total_images}"
                )

            if local_count > 0:
                st.success(
                    f"✅ {local_count} images available locally for faster loading!"
                )
            else:
                st.info(
                    "💡 Download images locally for faster loading using: `python3 download_all_ev_images.py`"
                )
            # Image display options
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.info(f"📸 Found {total_images} images in filtered data")

            with col2:
                images_per_page = st.selectbox(
                    "Images per page", [6, 12, 24, 48, 96], index=2  # Default to 24
                )

            with col3:
                images_per_row = st.selectbox(
                    "Images per row", [2, 3, 4, 6], index=1  # Default to 3
                )

            # Calculate pagination
            total_pages = (total_images - 1) // images_per_page + 1

            # Page selector
            if total_pages > 1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    current_page = st.selectbox(
                        f"Page (1 of {total_pages})", range(1, total_pages + 1), index=0
                    )
            else:
                current_page = 1

            # Calculate start and end indices for current page
            start_idx = (current_page - 1) * images_per_page
            end_idx = min(start_idx + images_per_page, total_images)

            # Get current page data
            current_page_data = all_image_data.iloc[start_idx:end_idx]

            # Display page info
            st.subheader(
                f"Page {current_page} of {total_pages} - Showing images {start_idx + 1} to {end_idx}"
            )

            # Progress bar for loading
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Display images in grid
            for i in range(0, len(current_page_data), images_per_row):
                cols = st.columns(images_per_row)

                for j, col in enumerate(cols):
                    if i + j < len(current_page_data):
                        row = current_page_data.iloc[i + j]

                        # Update progress
                        progress = (i + j + 1) / len(current_page_data)
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Loading image {i + j + 1} of {len(current_page_data)}..."
                        )

                        with col:
                            # Try to load image (local first, then URL)
                            image, source = load_image_local_or_url(
                                row["ad_archive_id"], row["new_image_url"]
                            )

                            if image:
                                # Show source indicator
                                source_icon = "💾" if source == "local" else "🌐"
                                caption = f"{source_icon} {row['page_name']} - {row['matched_car_models']}"

                                st.image(
                                    image,
                                    caption=caption,
                                    use_container_width=True,
                                )
                            else:
                                st.error("❌ Could not load image")
                                st.write(f"URL: {row['new_image_url'][:50]}...")

                            # Show AI insights prominently
                            if (
                                "ad_theme" in row
                                and pd.notna(row["ad_theme"])
                                and str(row["ad_theme"]).strip()
                            ):
                                st.markdown("**🎨 Image Theme:**")
                                theme_text = str(row["ad_theme"])
                                st.info(theme_text)

                            # Show OpenAI summary prominently
                            if (
                                "openai_summary" in row
                                and pd.notna(row["openai_summary"])
                                and str(row["openai_summary"]).strip()
                            ):
                                with st.expander(
                                    "🤖 AI Analysis Summary", expanded=True
                                ):
                                    summary_text = str(row["openai_summary"])
                                    st.markdown(summary_text)

                            # Show ad details in expander
                            with st.expander("📋 Ad Details"):
                                st.write(f"**🏢 Advertiser:** {row['page_name']}")
                                st.write(f"**🚗 Model:** {row['matched_car_models']}")
                                st.write(f"**🎯 CTA:** {row['cta_text']}")
                                st.write(f"**📝 Title:** {row['ad_title']}")
                                if (
                                    pd.notna(row["ad_text"])
                                    and str(row["ad_text"]).strip()
                                ):
                                    ad_text = str(row["ad_text"])
                                    if len(ad_text) > 200:
                                        st.write(f"**📄 Text:** {ad_text[:200]}...")
                                        with st.expander("Show full text"):
                                            st.write(ad_text)
                                    else:
                                        st.write(f"**📄 Text:** {ad_text}")

                                # Additional metadata
                                if "page_classification" in row and pd.notna(
                                    row["page_classification"]
                                ):
                                    st.write(
                                        f"**🏷️ Type:** {row['page_classification']}"
                                    )
                                if "start_date" in row and pd.notna(row["start_date"]):
                                    st.write(f"**📅 Start:** {row['start_date']}")
                                if (
                                    "male_percentage" in row
                                    and pd.notna(row["male_percentage"])
                                    and row["male_percentage"] > 0
                                ):
                                    st.write(
                                        f"**👥 Targeting:** {row['male_percentage']:.1f}% M, {row['female_percentage']:.1f}% F"
                                    )

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Show pagination info
            if total_pages > 1:
                st.markdown(
                    f"**Page {current_page} of {total_pages}** | **Total Images: {total_images}**"
                )

    with tab5:
        st.header("📈 Advanced Analytics")

        # Time series analysis
        if "start_date" in filtered_df.columns:
            st.subheader("Ad Campaign Timeline")

            # Convert dates
            filtered_df["start_date_parsed"] = pd.to_datetime(
                filtered_df["start_date"], errors="coerce"
            )

            if filtered_df["start_date_parsed"].notna().any():
                # Group by month and car model
                monthly_data = (
                    filtered_df.groupby(
                        [
                            filtered_df["start_date_parsed"].dt.to_period("M"),
                            "matched_car_models",
                        ]
                    )
                    .size()
                    .reset_index(name="ad_count")
                )

                monthly_data["month"] = monthly_data["start_date_parsed"].astype(str)

                fig = px.line(
                    monthly_data,
                    x="month",
                    y="ad_count",
                    color="matched_car_models",
                    title="Ad Volume Over Time by Car Model",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Correlation analysis
        st.subheader("Targeting Correlations")

        numeric_cols = [
            "male_percentage",
            "female_percentage",
            "total_male_audience",
            "total_female_audience",
            "page_like_count",
        ]
        available_numeric_cols = [
            col for col in numeric_cols if col in filtered_df.columns
        ]

        if len(available_numeric_cols) >= 2:
            corr_data = filtered_df[available_numeric_cols].corr()

            fig = px.imshow(
                corr_data,
                title="Correlation Matrix of Targeting Metrics",
                color_continuous_scale="RdBu",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Raw data export
        st.subheader("📥 Data Export")

        if st.button("Download Filtered Data as CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ev_ads_filtered_{len(filtered_df)}_records.csv",
                mime="text/csv",
            )

    with tab6:
        st.header("🎨 Theme Analysis")
        st.markdown(
            "📊 **Pre-computed theme analysis results** - View advertising themes and messaging patterns"
        )

        # Load theme analysis data
        theme_data = load_theme_analysis_data()

        # Display file loading info
        files_loaded = display_theme_files_info(theme_data)

        if files_loaded > 0:
            # Display pre-computed theme analysis

            # Create tabs for different analyses
            theme_tab1, theme_tab2, theme_tab3 = st.tabs(
                ["📊 Overall Themes", "🚗 Themes by Model", "🔍 NLP Keywords"]
            )

            with theme_tab1:
                st.subheader("Overall Theme Frequency")

                if "overall" in theme_data:
                    overall_themes = theme_data["overall"]

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        fig = px.bar(
                            overall_themes.head(15),
                            x="frequency",
                            y="theme",
                            orientation="h",
                            title="Top 15 Themes Across All EV Ads",
                            labels={
                                "frequency": "Number of Mentions",
                                "theme": "Theme",
                            },
                        )
                        fig.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("📈 Key Statistics")
                        total_mentions = overall_themes["frequency"].sum()
                        st.metric("Total Theme Mentions", f"{total_mentions:,}")
                        st.metric("Unique Themes", len(overall_themes))

                        top_theme = overall_themes.iloc[0]
                        st.metric(
                            "Top Theme",
                            top_theme["theme"],
                            f"{top_theme['percentage']:.1f}%",
                        )

                        # Show top 10 themes as metrics
                        st.subheader("🏆 Top 10 Themes")
                        for _, row in overall_themes.head(10).iterrows():
                            st.write(
                                f"**{row['theme']}**: {row['frequency']:,} ({row['percentage']:.1f}%)"
                            )

            with theme_tab2:
                st.subheader("Themes by Car Model")

                if "by_model" in theme_data:
                    model_themes = theme_data["by_model"]

                    # Get top models by total theme mentions
                    model_totals = (
                        model_themes.groupby("car_model")["frequency"]
                        .sum()
                        .sort_values(ascending=False)
                    )

                    # Model selector
                    selected_model = st.selectbox(
                        "Select Car Model for Detailed Analysis:",
                        options=model_totals.index.tolist(),
                        index=0,
                    )

                    if selected_model:
                        model_data = model_themes[
                            model_themes["car_model"] == selected_model
                        ].sort_values("frequency", ascending=False)

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            fig = px.bar(
                                model_data.head(10),
                                x="frequency",
                                y="theme",
                                orientation="h",
                                title=f"Top 10 Themes for {selected_model}",
                                labels={
                                    "frequency": "Number of Mentions",
                                    "theme": "Theme",
                                },
                            )
                            fig.update_layout(
                                yaxis={"categoryorder": "total ascending"}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.subheader(f"📊 {selected_model} Stats")
                            total_mentions = model_data["frequency"].sum()
                            st.metric("Total Mentions", f"{total_mentions:,}")
                            st.metric("Unique Themes", len(model_data))

                            if len(model_data) > 0:
                                top_theme = model_data.iloc[0]
                                st.metric(
                                    "Top Theme",
                                    top_theme["theme"],
                                    f"{top_theme['percentage']:.1f}%",
                                )

                    # Comparison heatmap
                    st.subheader("🔥 Theme Heatmap by Model")

                    # Get top 5 models and top 10 themes for heatmap
                    top_models = model_totals.head(5).index.tolist()
                    top_themes_overall = (
                        theme_data["overall"].head(10)["theme"].tolist()
                    )

                    # Create pivot table
                    heatmap_data = (
                        model_themes[
                            (model_themes["car_model"].isin(top_models))
                            & (model_themes["theme"].isin(top_themes_overall))
                        ]
                        .pivot(index="theme", columns="car_model", values="percentage")
                        .fillna(0)
                    )

                    if not heatmap_data.empty:
                        fig = px.imshow(
                            heatmap_data,
                            title="Theme Distribution Heatmap (Top 5 Models vs Top 10 Themes)",
                            labels={"color": "Percentage of Model's Themes"},
                            color_continuous_scale="Viridis",
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with theme_tab3:
                st.subheader("NLP-Discovered Keywords & Themes")

                if "nlp_keywords" in theme_data:
                    nlp_keywords = theme_data["nlp_keywords"]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("🔍 Top Keywords (TF-IDF)")

                        fig = px.bar(
                            nlp_keywords.head(20),
                            x="tfidf_score",
                            y="keyword",
                            orientation="h",
                            title="Top 20 Keywords by TF-IDF Score",
                        )
                        fig.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("📈 Keyword Statistics")
                        st.metric("Total Keywords", len(nlp_keywords))

                        top_keyword = nlp_keywords.iloc[0]
                        st.metric(
                            "Top Keyword",
                            top_keyword["keyword"],
                            f"TF-IDF: {top_keyword['tfidf_score']:.4f}",
                        )

                        # Show top keywords list
                        st.subheader("🏆 Top 15 Keywords")
                        for _, row in nlp_keywords.head(15).iterrows():
                            st.write(f"**{row['keyword']}**: {row['tfidf_score']:.4f}")

                if "nlp_themes" in theme_data:
                    st.subheader("🎨 NLP-Discovered Themes")
                    nlp_themes = theme_data["nlp_themes"]

                    for _, theme in nlp_themes.iterrows():
                        with st.expander(f"🎯 {theme['theme_name']}"):
                            st.write(f"**Keywords:** {theme['top_keywords']}")
                            st.write(f"**Cluster ID:** {theme['cluster_id']}")


if __name__ == "__main__":
    main()
