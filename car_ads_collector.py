import os
import json
import time
from datetime import datetime
from apify_client import ApifyClient


def main():
    """
    Collect ads for 14 car models across Meta, TikTok, YouTube, and Google
    Focus on EU markets for German automotive client
    """

    # API setup
    api_token = os.getenv(
        "APIFY_TOKEN", "apify_api_hTwjqJT3c9h9Qeg5SgcvUg4TqqPeQc20GfY3"
    )
    client = ApifyClient(api_token)

    # Complete EV model list - all 16 models
    car_models = [
        "Cupra Born",
        "Mini Aceman E",
        "Volkswagen ID3",
        "Kia EV3",
        "Volvo EX30",
        "Renault Megane E-Tech",
        "Volvo EC40",
        "Volkswagen ID5",
        "MG4",
        "BMW iX2",
        "Hyundai Ioniq 5",
        "Volkswagen ID.4",
        "Audi Q4 e-tron",
        "BMW iX1",
        "BMW iX3",
        "Tesla Model Y",
    ]

    # Model-specific keywords for precise matching (brand + model required)
    model_keywords = {
        "Cupra Born": {"brand": ["cupra", "seat"], "model": ["born"]},
        "Mini Aceman E": {"brand": ["mini"], "model": ["aceman"]},
        "Volkswagen ID3": {"brand": ["volkswagen", "vw"], "model": ["id3", "id.3"]},
        "Kia EV3": {"brand": ["kia"], "model": ["ev3", "ev.3"]},
        "Volvo EX30": {"brand": ["volvo"], "model": ["ex30", "ex.30"]},
        "Renault Megane E-Tech": {
            "brand": ["renault"],
            "model": ["megane", "e-tech", "etech"],
        },
        "Volvo EC40": {"brand": ["volvo"], "model": ["ec40", "ec.40"]},
        "Volkswagen ID5": {"brand": ["volkswagen", "vw"], "model": ["id5", "id.5"]},
        "MG4": {"brand": ["mg", "mgmotor"], "model": ["mg4"]},
        "BMW iX2": {"brand": ["bmw"], "model": ["ix2", "i-x2"]},
        "Hyundai Ioniq 5": {"brand": ["hyundai"], "model": ["ioniq", "ioniq5"]},
        "Volkswagen ID.4": {"brand": ["volkswagen", "vw"], "model": ["id4", "id.4"]},
        "Audi Q4 e-tron": {"brand": ["audi"], "model": ["q4", "e-tron", "etron"]},
        "BMW iX1": {"brand": ["bmw"], "model": ["ix1", "i-x1"]},
        "BMW iX3": {"brand": ["bmw"], "model": ["ix3", "i-x3"]},
        "Tesla Model Y": {"brand": ["tesla"], "model": ["model y", "modely"]},
    }

    print("🚗 Starting Global Car Ad Collection for German Automotive Client")
    print(f"📊 Collecting ads for {len(car_models)} car models globally")
    print("🎯 Platforms: Meta (Facebook), TikTok, YouTube, Google Ads")
    print("💡 Using scrapeAdDetails=true for maximum data extraction")

    # Results storage
    all_results = {
        "meta_ads": [],
        "collection_info": {
            "timestamp": datetime.now().isoformat(),
            "car_models": car_models,
            "collection_type": "global_no_country_limit",
            "total_searches": 0,
        },
    }

    # === META ADS COLLECTION ===
    print("\n🔵 Starting Meta (Facebook) Ads Collection...")

    # Enhanced search strategy using model-specific keywords
    meta_urls = []
    # Track mapping between URLs and the keywords that generated them
    url_to_keyword_mapping = {}

    for car in car_models[:3]:
        if car not in model_keywords:
            continue

        keywords = model_keywords[car]
        brands = keywords["brand"]
        models = keywords["model"]

        # Strategy: Create comprehensive search terms for each model
        base_url = "https://www.facebook.com/ads/library/"

        # Generate search terms combining brands and models
        search_terms = []

        # 1. Exact model name in quotes
        search_terms.append(f'"{car}"')

        # 2. Brand + model combinations
        for brand in brands:
            for model in models:
                search_terms.append(f"{brand} {model}")
                search_terms.append(f'"{brand} {model}"')  # Quoted version

        # 3. Brand + electric (for EV context)
        for brand in brands:
            search_terms.append(f"{brand} electric")
            search_terms.append(f"{brand} EV")

        print(f"\n🔍 Searching for {car}:")
        print(f"   • Search terms: {len(search_terms)}")
        print(f"   • Brands: {', '.join(brands)}")
        print(f"   • Models: {', '.join(models)}")

        for search_term in search_terms:
            params = {
                "active_status": "all",
                "ad_type": "all",
                "q": search_term,
                "search_type": (
                    "keyword_exact" if '"' in search_term else "keyword_unordered"
                ),
                "media_type": "all",
            }

            # Fix f-string backslash issue
            encoded_params = []
            for k, v in params.items():
                encoded_v = v.replace(" ", "%20").replace('"', "%22")
                encoded_params.append(f"{k}={encoded_v}")
            url_params = "&".join(encoded_params)
            full_url = f"{base_url}?{url_params}"

            # Store URL with metadata for tracking
            url_entry = {
                "url": full_url,
                "search_term": search_term,
                "target_model": car,
                "search_type": params["search_type"],
            }
            meta_urls.append(url_entry)

            # Create mapping for later reference
            url_to_keyword_mapping[full_url] = {
                "search_term": search_term,
                "target_model": car,
                "search_type": params["search_type"],
                "brands": brands,
                "models": models,
            }

    print(f"🎯 Generated {len(meta_urls)} targeted search URLs for better precision")

    # Single large request - more cost efficient!
    print(f"📦 Processing all {len(meta_urls)} URLs in one efficient request...")

    run_input = {
        "urls": meta_urls,  # All URLs at once
        "scrapeAdDetails": True,  # Get maximum data including EU transparency
        "scrapePageAds.activeStatus": "all",
        "period": "",
        # No count limit - get ALL available ads
    }

    try:
        # Run Facebook Ads Library scraper - single large request
        print("🚀 Starting single large Meta ads collection (most cost-efficient)...")
        run = client.actor("curious_coder/facebook-ads-library-scraper").call(
            run_input=run_input
        )

        print(f"✅ Meta ads collection completed!")
        print(
            f"💾 Dataset: https://console.apify.com/storage/datasets/{run['defaultDatasetId']}"
        )

        # Collect and filter results
        print("📥 Downloading results...")
        raw_ads = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            # Add keyword mapping information to each ad
            source_url = item.get("url", "")
            if source_url in url_to_keyword_mapping:
                item["keyword_mapping"] = url_to_keyword_mapping[source_url]
            raw_ads.append(item)

        print(f"📊 Raw ads collected: {len(raw_ads)}")
        print("🔍 Matching ads to specific EV models...")

        # Process ads with model-specific matching
        model_matched_ads = []
        ads_by_model = {model: [] for model in car_models}
        ads_by_keyword = {}  # Track which ads came from which search terms

        irrelevant_keywords = [
            "rental",
            "rent",
            "hire",
            "marketplace",
            "used car",
            "second hand",
            "car dealer",
            "auto dealer",
            "booking",
            "reservation",
        ]

        for ad in raw_ads:
            # Extract ad text content
            ad_text = ""
            if ad.get("snapshot", {}).get("body", {}).get("text"):
                ad_text += ad["snapshot"]["body"]["text"].lower() + " "

            page_name = ad.get("page_name", "").lower()
            ad_text += page_name + " "

            # Skip obvious irrelevant ads
            is_irrelevant = any(keyword in ad_text for keyword in irrelevant_keywords)
            if is_irrelevant:
                continue

            # Track which search term generated this ad
            search_term_info = ad.get("keyword_mapping", {})
            if search_term_info:
                search_term = search_term_info.get("search_term", "unknown")
                target_model = search_term_info.get("target_model", "unknown")

                # Initialize keyword tracking
                if search_term not in ads_by_keyword:
                    ads_by_keyword[search_term] = []
                ads_by_keyword[search_term].append(ad)

            # Try to match to specific EV models using brand + model logic
            matched_models = []
            for model, keyword_groups in model_keywords.items():
                brand_match = any(
                    brand.lower() in ad_text for brand in keyword_groups["brand"]
                )
                model_match = any(
                    model_kw.lower() in ad_text for model_kw in keyword_groups["model"]
                )

                # Require BOTH brand and model keywords
                if brand_match and model_match:
                    matched_brands = [
                        b for b in keyword_groups["brand"] if b.lower() in ad_text
                    ]
                    matched_model_kws = [
                        m for m in keyword_groups["model"] if m.lower() in ad_text
                    ]

                    # Add matching info to ad
                    ad["matched_model"] = model
                    ad["matched_brand_keywords"] = matched_brands
                    ad["matched_model_keywords"] = matched_model_kws

                    ads_by_model[model].append(ad)
                    matched_models.append(model)

            # Add to results if matched any model
            if matched_models:
                ad["all_matched_models"] = matched_models
                model_matched_ads.append(ad)

        # Summary by model
        print(f"\n📈 Model-specific matching results:")
        total_matched = 0
        for model in car_models:
            count = len(ads_by_model[model])
            total_matched += count
            if count > 0:
                print(f"   🚗 {model}: {count} ads")
            else:
                print(f"   ⚪ {model}: No ads found")

        print(f"\n📊 Total matched ads: {total_matched}")
        print(f"📊 Filtered out: {len(raw_ads) - total_matched} irrelevant ads")

        # Store results with model-specific data and keyword mapping
        all_results["meta_ads"] = model_matched_ads
        all_results["ads_by_model"] = ads_by_model
        all_results["ads_by_keyword"] = ads_by_keyword
        all_results["keyword_mapping"] = url_to_keyword_mapping
        all_results["collection_info"]["total_searches"] = len(meta_urls)
        all_results["collection_info"]["raw_ads_collected"] = len(raw_ads)
        all_results["collection_info"]["model_matched_ads"] = total_matched

        print(f"📈 Collection complete: {total_matched} model-specific ads found")

        # Show keyword mapping summary
        print(f"\n🔍 Keyword Mapping Summary:")
        print(f"📊 Total search terms used: {len(ads_by_keyword)}")

        # Show top performing keywords
        keyword_performance = [(k, len(v)) for k, v in ads_by_keyword.items()]
        keyword_performance.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🏆 Top 10 performing search terms:")
        for i, (keyword, count) in enumerate(keyword_performance[:10]):
            print(f"   {i+1}. '{keyword}': {count} ads")

        if len(keyword_performance) > 10:
            remaining = len(keyword_performance) - 10
            print(f"   ... and {remaining} more search terms")

    except Exception as e:
        print(f"❌ Error in Meta ads collection: {str(e)}")
        print("💡 If timeout occurs, we can split into 2-3 large batches instead")

    # === SAVE RESULTS ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"car_ads_collection_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Collection Complete!")
    print(f"📊 Total Meta ads collected: {len(all_results['meta_ads'])}")
    print(f"💾 Results saved to: {filename}")

    # Preview results
    if all_results["meta_ads"]:
        print(f"\n📋 Preview of first 3 Meta ads:")
        for i, ad in enumerate(all_results["meta_ads"][:3]):
            print(f"\n--- Ad {i+1} ---")

            # Show which keyword generated this ad
            if "keyword_mapping" in ad:
                mapping = ad["keyword_mapping"]
                search_term = mapping.get("search_term", "unknown")
                target_model = mapping.get("target_model", "unknown")
                print(
                    f"🔍 Generated by search: '{search_term}' (targeting: {target_model})"
                )

            for key in [
                "adText",
                "pageTitle",
                "impressions",
                "spend",
                "adCreativeText",
            ]:
                if key in ad and ad[key]:
                    value = (
                        str(ad[key])[:100] + "..."
                        if len(str(ad[key])) > 100
                        else ad[key]
                    )
                    print(f"{key}: {value}")

    print(f"\n🔍 Next Steps:")
    print(f"1. Review {filename} for Meta ads data")
    print(f"2. Set up TikTok, YouTube, and Google Ads collection")
    print(f"3. Analyze creative themes and performance metrics")
    print(f"4. Generate insights for German automotive client")


if __name__ == "__main__":
    main()
