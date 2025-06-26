#!/usr/bin/env python3
"""
Facebook Ads Library JSON to CSV Converter
Converts complex nested JSON data from Facebook Ads Library scraper to comprehensive CSV format.
"""

import json
import csv
import sys
from datetime import datetime
from typing import Any, Dict, List, Union


def safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """Safely get a value from dictionary with default fallback."""
    return data.get(key, default)


def flatten_list(lst: List, separator: str = "; ") -> str:
    """Convert list to string representation."""
    if not lst:
        return ""
    return separator.join(str(item) for item in lst)


def flatten_dict(data: Dict, prefix: str = "", separator: str = "_") -> Dict[str, Any]:
    """Flatten nested dictionary structure."""
    flattened = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, separator))
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                # Handle list of dictionaries (like images, videos)
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flattened.update(flatten_dict(item, f"{new_key}_{i}", separator))
                    else:
                        flattened[f"{new_key}_{i}"] = str(item)
            else:
                # Handle simple lists
                flattened[new_key] = flatten_list(value)
        else:
            flattened[new_key] = value
    
    return flattened


def convert_timestamp(timestamp: Union[int, str, None]) -> str:
    """Convert Unix timestamp to readable date format."""
    if timestamp is None:
        return ""
    try:
        if isinstance(timestamp, str):
            timestamp = int(timestamp)
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return str(timestamp)


def process_ad_record(record: Dict) -> Dict[str, Any]:
    """Process a single ad record and flatten all nested structures."""
    flattened = {}
    
    # Top-level fields
    flattened['ad_archive_id'] = safe_get(record, 'ad_archive_id')
    flattened['page_id'] = safe_get(record, 'page_id')
    flattened['is_active'] = safe_get(record, 'is_active')
    flattened['has_user_reported'] = safe_get(record, 'has_user_reported')
    flattened['report_count'] = safe_get(record, 'report_count')
    flattened['page_is_deleted'] = safe_get(record, 'page_is_deleted')
    flattened['page_name'] = safe_get(record, 'page_name')
    flattened['ad_id'] = safe_get(record, 'ad_id')
    flattened['collation_count'] = safe_get(record, 'collation_count')
    flattened['collation_id'] = safe_get(record, 'collation_id')
    flattened['contains_digital_created_media'] = safe_get(record, 'contains_digital_created_media')
    flattened['contains_sensitive_content'] = safe_get(record, 'contains_sensitive_content')
    flattened['currency'] = safe_get(record, 'currency')
    flattened['entity_type'] = safe_get(record, 'entity_type')
    flattened['gated_type'] = safe_get(record, 'gated_type')
    flattened['hidden_safety_data'] = safe_get(record, 'hidden_safety_data')
    flattened['hide_data_status'] = safe_get(record, 'hide_data_status')
    flattened['is_aaa_eligible'] = safe_get(record, 'is_aaa_eligible')
    flattened['is_profile_page'] = safe_get(record, 'is_profile_page')
    flattened['total'] = safe_get(record, 'total')
    flattened['url'] = safe_get(record, 'url')
    
    # Convert timestamps
    flattened['start_date'] = convert_timestamp(safe_get(record, 'start_date'))
    flattened['end_date'] = convert_timestamp(safe_get(record, 'end_date'))
    
    # Handle list fields
    flattened['menu_items'] = flatten_list(safe_get(record, 'menu_items', []))
    flattened['archive_types'] = flatten_list(safe_get(record, 'archive_types', []))
    flattened['categories'] = flatten_list(safe_get(record, 'categories', []))
    flattened['political_countries'] = flatten_list(safe_get(record, 'political_countries', []))
    flattened['publisher_platform'] = flatten_list(safe_get(record, 'publisher_platform', []))
    flattened['targeted_or_reached_countries'] = flatten_list(safe_get(record, 'targeted_or_reached_countries', []))
    flattened['violation_types'] = flatten_list(safe_get(record, 'violation_types', []))
    
    # Handle nested objects
    impressions = safe_get(record, 'impressions_with_index', {})
    if impressions:
        flattened.update(flatten_dict(impressions, 'impressions'))
    
    regional_data = safe_get(record, 'regional_regulation_data', {})
    if regional_data:
        flattened.update(flatten_dict(regional_data, 'regional_regulation'))
    
    advertiser = safe_get(record, 'advertiser', {})
    if advertiser:
        flattened.update(flatten_dict(advertiser, 'advertiser'))
    
    aaa_info = safe_get(record, 'aaa_info', {})
    if aaa_info:
        flattened.update(flatten_dict(aaa_info, 'aaa_info'))
    
    # Handle snapshot data (the most complex part)
    snapshot = safe_get(record, 'snapshot', {})
    if snapshot:
        # Basic snapshot fields
        flattened['snapshot_page_id'] = safe_get(snapshot, 'page_id')
        flattened['snapshot_page_is_deleted'] = safe_get(snapshot, 'page_is_deleted')
        flattened['snapshot_page_profile_uri'] = safe_get(snapshot, 'page_profile_uri')
        flattened['snapshot_page_name'] = safe_get(snapshot, 'page_name')
        flattened['snapshot_page_profile_picture_url'] = safe_get(snapshot, 'page_profile_picture_url')
        flattened['snapshot_is_reshared'] = safe_get(snapshot, 'is_reshared')
        flattened['snapshot_caption'] = safe_get(snapshot, 'caption')
        flattened['snapshot_cta_text'] = safe_get(snapshot, 'cta_text')
        flattened['snapshot_cta_type'] = safe_get(snapshot, 'cta_type')
        flattened['snapshot_country_iso_code'] = safe_get(snapshot, 'country_iso_code')
        flattened['snapshot_current_page_name'] = safe_get(snapshot, 'current_page_name')
        flattened['snapshot_display_format'] = safe_get(snapshot, 'display_format')
        flattened['snapshot_link_description'] = safe_get(snapshot, 'link_description')
        flattened['snapshot_link_url'] = safe_get(snapshot, 'link_url')
        flattened['snapshot_page_entity_type'] = safe_get(snapshot, 'page_entity_type')
        flattened['snapshot_page_is_profile_page'] = safe_get(snapshot, 'page_is_profile_page')
        flattened['snapshot_page_like_count'] = safe_get(snapshot, 'page_like_count')
        flattened['snapshot_title'] = safe_get(snapshot, 'title')
        flattened['snapshot_byline'] = safe_get(snapshot, 'byline')
        flattened['snapshot_disclaimer_label'] = safe_get(snapshot, 'disclaimer_label')
        flattened['snapshot_brazil_tax_id'] = safe_get(snapshot, 'brazil_tax_id')
        
        # Body text
        body = safe_get(snapshot, 'body', {})
        if body:
            flattened['snapshot_body_text'] = safe_get(body, 'text')
        
        # Page categories
        flattened['snapshot_page_categories'] = flatten_list(safe_get(snapshot, 'page_categories', []))
        
        # Images
        images = safe_get(snapshot, 'images', [])
        if images:
            for i, image in enumerate(images):
                flattened[f'image_{i}_original_url'] = safe_get(image, 'original_image_url')
                flattened[f'image_{i}_resized_url'] = safe_get(image, 'resized_image_url')
                flattened[f'image_{i}_watermarked_url'] = safe_get(image, 'watermarked_resized_image_url')
        
        # Videos
        videos = safe_get(snapshot, 'videos', [])
        if videos:
            for i, video in enumerate(videos):
                if isinstance(video, dict):
                    flattened.update(flatten_dict(video, f'video_{i}'))
        
        # Additional snapshot arrays
        flattened['snapshot_cards'] = flatten_list(safe_get(snapshot, 'cards', []))
        flattened['snapshot_ec_certificates'] = flatten_list(safe_get(snapshot, 'ec_certificates', []))
        flattened['snapshot_extra_images'] = flatten_list(safe_get(snapshot, 'extra_images', []))
        flattened['snapshot_extra_links'] = flatten_list(safe_get(snapshot, 'extra_links', []))
        flattened['snapshot_extra_texts'] = flatten_list(safe_get(snapshot, 'extra_texts', []))
        flattened['snapshot_extra_videos'] = flatten_list(safe_get(snapshot, 'extra_videos', []))
    
    return flattened


def convert_json_to_csv(json_file: str, csv_file: str):
    """Convert Facebook Ads Library JSON to comprehensive CSV."""
    print(f"Loading JSON data from {json_file}...")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    if not isinstance(data, list):
        print("Error: JSON data should be a list of ad records")
        return
    
    print(f"Processing {len(data)} ad records...")
    
    # Process all records
    processed_records = []
    for i, record in enumerate(data):
        if i % 1000 == 0:
            print(f"Processed {i} records...")
        
        try:
            flattened = process_ad_record(record)
            processed_records.append(flattened)
        except Exception as e:
            print(f"Error processing record {i}: {e}")
            continue
    
    if not processed_records:
        print("No records were successfully processed")
        return
    
    # Get all unique column names
    all_columns = set()
    for record in processed_records:
        all_columns.update(record.keys())
    
    columns = sorted(list(all_columns))
    
    print(f"Writing CSV with {len(columns)} columns to {csv_file}...")
    
    # Write to CSV
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for record in processed_records:
                # Fill missing columns with empty strings
                row = {col: record.get(col, '') for col in columns}
                writer.writerow(row)
        
        print(f"Successfully converted {len(processed_records)} records to {csv_file}")
        print(f"CSV contains {len(columns)} columns")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")


if __name__ == "__main__":
    input_file = "dataset_facebook-ads-library-scraper_2025-06-25_06-54-41-268.json"
    output_file = "facebook_ads_library_data.csv"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    convert_json_to_csv(input_file, output_file)
