#!/usr/bin/env python3
"""
Create a clean, manageable CSV with essential Facebook Ads Library columns
"""

import json
import csv
import pandas as pd
from datetime import datetime


def convert_timestamp(timestamp):
    """Convert Unix timestamp to readable date format."""
    if timestamp is None or timestamp == "":
        return ""
    try:
        if isinstance(timestamp, str):
            timestamp = int(timestamp)
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(timestamp)


def safe_get(data, key, default=""):
    """Safely get a value from dictionary with default fallback."""
    value = data.get(key, default)
    return value if value is not None else default


def flatten_list(lst, separator="; "):
    """Convert list to string representation."""
    if not lst:
        return ""
    return separator.join(str(item) for item in lst)


def extract_essential_data(record):
    """Extract only the most important columns for analysis."""
    data = {}

    # Basic ad information
    data["ad_archive_id"] = safe_get(record, "ad_archive_id")
    data["page_id"] = safe_get(record, "page_id")
    data["page_name"] = safe_get(record, "page_name")
    data["is_active"] = safe_get(record, "is_active")
    data["currency"] = safe_get(record, "currency")
    data["entity_type"] = safe_get(record, "entity_type")
    data["gated_type"] = safe_get(record, "gated_type")
    data["contains_digital_created_media"] = safe_get(
        record, "contains_digital_created_media"
    )
    data["contains_sensitive_content"] = safe_get(record, "contains_sensitive_content")
    data["hidden_safety_data"] = safe_get(record, "hidden_safety_data")

    # Dates
    data["start_date"] = convert_timestamp(safe_get(record, "start_date"))
    data["end_date"] = convert_timestamp(safe_get(record, "end_date"))

    # Categories and targeting
    data["categories"] = flatten_list(safe_get(record, "categories", []))
    data["publisher_platform"] = flatten_list(
        safe_get(record, "publisher_platform", [])
    )
    data["targeted_countries"] = flatten_list(
        safe_get(record, "targeted_or_reached_countries", [])
    )
    data["political_countries"] = flatten_list(
        safe_get(record, "political_countries", [])
    )
    data["archive_types"] = flatten_list(safe_get(record, "archive_types", []))
    data["violation_types"] = flatten_list(safe_get(record, "violation_types", []))

    # Metrics
    data["collation_count"] = safe_get(record, "collation_count")
    data["total"] = safe_get(record, "total")
    data["spend"] = safe_get(record, "spend")
    data["reach_estimate"] = safe_get(record, "reach_estimate")
    data["total_active_time"] = safe_get(record, "total_active_time")

    # Snapshot data (ad content)
    snapshot = safe_get(record, "snapshot", {})
    if snapshot:
        data["ad_title"] = safe_get(snapshot, "title")
        data["ad_caption"] = safe_get(snapshot, "caption")
        data["cta_text"] = safe_get(snapshot, "cta_text")
        data["cta_type"] = safe_get(snapshot, "cta_type")
        data["display_format"] = safe_get(snapshot, "display_format")
        data["link_url"] = safe_get(snapshot, "link_url")
        data["link_description"] = safe_get(snapshot, "link_description")
        data["page_like_count"] = safe_get(snapshot, "page_like_count")
        data["page_categories"] = flatten_list(
            safe_get(snapshot, "page_categories", [])
        )
        data["page_profile_uri"] = safe_get(snapshot, "page_profile_uri")

        # Ad text content
        body = safe_get(snapshot, "body", {})
        data["ad_text"] = safe_get(body, "text") if body else ""

        # Images
        images = safe_get(snapshot, "images", [])
        if images:
            data["image_count"] = len(images)
            data["first_image_url"] = (
                safe_get(images[0], "original_image_url") if images else ""
            )
        else:
            data["image_count"] = 0
            data["first_image_url"] = ""

        # Videos
        videos = safe_get(snapshot, "videos", [])
        data["video_count"] = len(videos) if videos else 0

    # Advertiser info
    advertiser = safe_get(record, "advertiser", {})
    if advertiser:
        data["advertiser_name"] = safe_get(advertiser, "name")
        data["advertiser_id"] = safe_get(advertiser, "id")

    # Impressions data (simplified)
    impressions = safe_get(record, "impressions_with_index", {})
    if impressions:
        data["impressions_lower_bound"] = safe_get(impressions, "lower_bound")
        data["impressions_upper_bound"] = safe_get(impressions, "upper_bound")
        data["impressions_text"] = safe_get(impressions, "impressions_text")

    # AAA (Audience Analytics & Advertising) data - Gender and demographic breakdown
    aaa_info = safe_get(record, "aaa_info", {})
    if aaa_info:
        # Age audience data
        data["age_audience_min"] = safe_get(aaa_info, "age_audience_min")
        data["age_audience_max"] = safe_get(aaa_info, "age_audience_max")
        data["age_audience"] = safe_get(aaa_info, "age_audience")

        # Gender breakdown - aggregate across all countries
        total_male = 0
        total_female = 0
        total_unknown = 0
        countries_with_data = []

        # Extract gender data from age_country_gender_reach_breakdown
        breakdown_data = safe_get(aaa_info, "age_country_gender_reach_breakdown", [])
        if breakdown_data:
            for country_data in breakdown_data:
                if isinstance(country_data, dict):
                    country_code = safe_get(country_data, "country_code")
                    if country_code:
                        countries_with_data.append(country_code)

                    # Sum up gender data across age groups
                    age_gender_breakdowns = safe_get(
                        country_data, "age_gender_breakdowns", []
                    )
                    for age_group in age_gender_breakdowns:
                        if isinstance(age_group, dict):
                            total_male += safe_get(age_group, "male", 0) or 0
                            total_female += safe_get(age_group, "female", 0) or 0
                            total_unknown += safe_get(age_group, "unknown", 0) or 0

        data["total_male_audience"] = total_male
        data["total_female_audience"] = total_female
        data["total_unknown_gender"] = total_unknown
        data["countries_with_demographic_data"] = flatten_list(countries_with_data)

        # Calculate gender percentages
        total_gendered = total_male + total_female + total_unknown
        if total_gendered > 0:
            data["male_percentage"] = round((total_male / total_gendered) * 100, 2)
            data["female_percentage"] = round((total_female / total_gendered) * 100, 2)
            data["unknown_gender_percentage"] = round(
                (total_unknown / total_gendered) * 100, 2
            )
        else:
            data["male_percentage"] = 0
            data["female_percentage"] = 0
            data["unknown_gender_percentage"] = 0

        # Additional AAA metrics
        data["delivery_by_region"] = flatten_list(
            safe_get(aaa_info, "delivery_by_region", [])
        )
        data["demographic_distribution"] = flatten_list(
            safe_get(aaa_info, "demographic_distribution", [])
        )

        # Potential reach data
        potential_reach = safe_get(aaa_info, "potential_reach", {})
        if potential_reach:
            data["potential_reach_lower"] = safe_get(potential_reach, "lower_bound")
            data["potential_reach_upper"] = safe_get(potential_reach, "upper_bound")

    # Regional regulation data
    regional_data = safe_get(record, "regional_regulation_data", {})
    if regional_data:
        data["is_finserv_regulated"] = safe_get(
            regional_data, "finserv_is_deemed_finserv"
        )
        data["is_limited_delivery"] = safe_get(
            regional_data, "finserv_is_limited_delivery"
        )
        data["anti_scam_limited"] = safe_get(
            regional_data, "tw_anti_scam_is_limited_delivery"
        )

    return data


def create_clean_csv():
    """Create a clean CSV with essential columns only."""
    print("Loading JSON data...")

    try:
        with open(
            "dataset_facebook-ads-library-scraper_2025-06-25_06-54-41-268.json",
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    print(f"Processing {len(data)} records...")

    # Extract essential data
    clean_records = []
    for i, record in enumerate(data):
        if i % 1000 == 0:
            print(f"Processed {i} records...")

        try:
            clean_data = extract_essential_data(record)
            clean_records.append(clean_data)
        except Exception as e:
            print(f"Error processing record {i}: {e}")
            continue

    if not clean_records:
        print("No records were successfully processed")
        return

    # Define column order
    columns = [
        # Basic ad information
        "ad_archive_id",
        "page_id",
        "page_name",
        "advertiser_name",
        "advertiser_id",
        "ad_title",
        "ad_text",
        "ad_caption",
        "cta_text",
        "cta_type",
        "display_format",
        "link_url",
        "link_description",
        # Dates and status
        "start_date",
        "end_date",
        "is_active",
        "currency",
        "entity_type",
        "gated_type",
        # Content flags
        "contains_digital_created_media",
        "contains_sensitive_content",
        "hidden_safety_data",
        # Categories and targeting
        "categories",
        "page_categories",
        "publisher_platform",
        "archive_types",
        "targeted_countries",
        "political_countries",
        "violation_types",
        # Page metrics
        "page_like_count",
        "page_profile_uri",
        "collation_count",
        "total",
        # Spend and reach
        "spend",
        "reach_estimate",
        "total_active_time",
        # Media content
        "image_count",
        "video_count",
        "first_image_url",
        # Impressions
        "impressions_lower_bound",
        "impressions_upper_bound",
        "impressions_text",
        # Gender and demographic breakdown
        "age_audience_min",
        "age_audience_max",
        "age_audience",
        "total_male_audience",
        "total_female_audience",
        "total_unknown_gender",
        "male_percentage",
        "female_percentage",
        "unknown_gender_percentage",
        "countries_with_demographic_data",
        # Additional AAA metrics
        "delivery_by_region",
        "demographic_distribution",
        "potential_reach_lower",
        "potential_reach_upper",
        # Regulatory data
        "is_finserv_regulated",
        "is_limited_delivery",
        "anti_scam_limited",
    ]

    print(f"Writing clean CSV with {len(columns)} essential columns...")

    # Write to CSV
    try:
        with open("facebook_ads_clean.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for record in clean_records:
                # Fill missing columns with empty strings
                row = {col: record.get(col, "") for col in columns}
                writer.writerow(row)

        print(
            f"Successfully created facebook_ads_clean.csv with {len(clean_records)} records"
        )
        print(f"Clean CSV contains {len(columns)} essential columns")

        # Show file size
        import os

        size_mb = os.path.getsize("facebook_ads_clean.csv") / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")

    except Exception as e:
        print(f"Error writing CSV file: {e}")


if __name__ == "__main__":
    create_clean_csv()
