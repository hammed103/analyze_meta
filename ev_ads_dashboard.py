#!/usr/bin/env python3
"""
Electric Vehicle Ads Analysis Dashboard
Interactive Streamlit dashboard for analyzing Facebook EV ads data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
import requests
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Meta Analysis - EV Ads Dashboard",
    page_icon="🚗⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    """Load and cache the EV ads data."""
    try:
        # Try to load the ultimate AI enhanced version first, then fall back to other versions
        try:
            df = pd.read_csv("facebook_ads_electric_vehicles_ultimate_ai_enhanced.csv")
            st.info(
                "✓ Loaded ultimate AI enhanced data with OpenAI summaries and image themes"
            )
        except FileNotFoundError:
            try:
                df = pd.read_csv(
                    "facebook_ads_electric_vehicles_with_openai_summaries.csv"
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
            "first_image_url": None,
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


def main():
    st.title("🚗⚡ Meta Analysis - Electric Vehicle Ads Dashboard")
    st.markdown("---")

    # Load data
    df = load_data()
    if df is None:
        return

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Car model filter
    car_models = sorted(df["matched_car_models"].unique())
    selected_models = st.sidebar.multiselect(
        "Select Car Models",
        car_models,
        default=car_models,  # Select ALL car models by default
    )

    # Advertiser type filter
    advertiser_types = sorted(df["page_classification"].unique())
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview",
            "🚗 Car Model Analysis",
            "🎯 Ad Creative Analysis",
            "🖼️ Image Gallery",
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
        all_image_data = filtered_df[filtered_df["first_image_url"].notna()].copy()
        total_images = len(all_image_data)

        if total_images == 0:
            st.info("No images found in the filtered data.")
        else:
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
                            # Try to load and display image
                            image = load_image_from_url(row["first_image_url"])

                            if image:
                                st.image(
                                    image,
                                    caption=f"{row['page_name']} - {row['matched_car_models']}",
                                    use_container_width=True,
                                )
                            else:
                                st.error("❌ Could not load image")
                                st.write(f"URL: {row['first_image_url'][:50]}...")

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


if __name__ == "__main__":
    main()
