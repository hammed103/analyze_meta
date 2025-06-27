#!/usr/bin/env python3
"""
Car Features Analyzer - AI model to assess car adverts for key features
Analyzes all text fields in the CSV to extract and categorize car features
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import json
from datetime import datetime


class CarFeaturesAnalyzer:
    def __init__(self):
        """Initialize the analyzer with feature keywords."""
        self.feature_keywords = {
            'range_and_charging': {
                'range': [
                    'WLTP range', 'miles per charge', 'official range', 'real-world range',
                    'battery capacity', 'usable battery', 'maximum range', 'range per kWh',
                    'mile range', 'km range', 'battery range', 'driving range', 'EPA range'
                ],
                'charging': [
                    'rapid charging', 'fast charging', 'DC charging', 'AC charging',
                    'charge time', '80% in', 'Type 2 connector', 'CCS connector',
                    'home charging', 'public charging', 'charging speed', 'kW charging',
                    'charging stations', 'Vehicle-to-Load', 'V2L', 'supercharging',
                    'charging port', 'plug-in', 'wall charger', 'charging cable'
                ]
            },
            'performance': {
                'acceleration': [
                    '0-62 mph', '0-60 mph', 'top speed', 'bhp', 'PS', 'kW power',
                    'torque', 'instant torque', 'dual motor', 'all-wheel drive',
                    'rear-wheel drive', 'N Grin Boost', 'Performance model',
                    'boost function', 'acceleration', 'horsepower', 'motor power'
                ],
                'handling': [
                    'driving dynamics', 'ride quality', 'adaptive drive mode',
                    'steering', 'suspension', 'braking', 'regenerative braking',
                    'motor output', 'sport mode', 'eco mode', 'comfort mode',
                    'driving modes', 'handling', 'stability'
                ]
            },
            'interior_and_comfort': {
                'space': [
                    'spacious', 'interior volume', 'headroom', 'legroom',
                    'flat floor', 'sliding center console', 'seating capacity',
                    'boot space', 'cargo space', 'folding rear seats',
                    'front trunk', 'frunk', 'storage space', 'passenger space'
                ],
                'comfort': [
                    'heated seats', 'ventilated seats', 'power-adjustable seats',
                    'premium upholstery', 'leather seats', 'ambient lighting',
                    'climate control', 'multi-zone climate', 'panoramic glass roof',
                    'luxurious', 'ergonomic', 'adjustable steering wheel',
                    'massage seats', 'memory seats', 'comfort features'
                ]
            },
            'infotainment_and_audio': {
                'infotainment': [
                    'touchscreen', 'digital cockpit', 'driver display',
                    'navigation', 'Apple CarPlay', 'Android Auto',
                    'voice assistant', 'OTA updates', 'navigation system',
                    'split-screen', 'Bluetooth', 'USB ports', 'infotainment',
                    'display screen', 'multimedia', 'connectivity'
                ],
                'audio': [
                    'sound system', 'Harman Kardon', 'premium audio',
                    'subwoofer', 'loudspeakers', 'audio settings',
                    'media streaming', 'Spotify', 'YouTube', 'Netflix',
                    'audio system', 'speakers', 'surround sound'
                ]
            },
            'exterior_design': {
                'design': [
                    'LED headlights', 'light strip', 'aerodynamic',
                    'modern styling', 'sleek lines', 'distinctive design',
                    'panoramic roof', 'alloy wheels', 'flush door handles',
                    'color options', 'sporty', 'SUV', 'crossover',
                    'roof rails', 'design language', 'exterior styling',
                    'body style', 'wheel design', 'paint options'
                ]
            },
            'safety_and_assistance': {
                'safety': [
                    'Euro NCAP', 'safety rating', 'airbags', 'crash protection',
                    'child safety', 'structural safety', 'safety features',
                    'protection systems', 'safety technology', '5-star safety'
                ],
                'assistance': [
                    'driver assistance', 'adaptive cruise control',
                    'lane keeping assist', 'blind spot monitoring',
                    'emergency braking', 'parking sensors', '360-degree camera',
                    'autopilot', 'autonomous driving', 'traffic sign recognition',
                    'collision warning', 'automatic emergency braking',
                    'driver aids', 'assistance systems', 'self-parking'
                ]
            },
            'connectivity_and_digital': {
                'connectivity': [
                    'wireless charging', 'Wi-Fi hotspot', 'Bluetooth',
                    'smartphone integration', 'remote app', 'over-the-air updates',
                    'digital key', 'cloud services', 'connected services',
                    'real-time traffic', 'remote climate control',
                    'digital assistant', 'navigation updates', 'vehicle-to-grid',
                    'app control', 'smart features', 'digital services'
                ]
            }
        }
        
        # Compile regex patterns for efficient matching
        self.compiled_patterns = {}
        for category, subcategories in self.feature_keywords.items():
            self.compiled_patterns[category] = {}
            for subcategory, keywords in subcategories.items():
                # Create case-insensitive regex patterns
                patterns = [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) 
                           for keyword in keywords]
                self.compiled_patterns[category][subcategory] = patterns

    def analyze_text(self, text):
        """Analyze a single text string for car features."""
        if pd.isna(text) or not isinstance(text, str):
            return {}
        
        results = {}
        text_lower = text.lower()
        
        for category, subcategories in self.compiled_patterns.items():
            results[category] = {}
            for subcategory, patterns in subcategories.items():
                matches = []
                for pattern in patterns:
                    found = pattern.findall(text)
                    matches.extend(found)
                
                results[category][subcategory] = {
                    'count': len(matches),
                    'matches': list(set(matches)) if matches else [],
                    'has_features': len(matches) > 0
                }
        
        return results

    def analyze_ad_record(self, row):
        """Analyze all text fields in a single ad record."""
        # Define text fields to analyze
        text_fields = [
            'ad_title', 'ad_text', 'cta_text', 'page_name',
            'extracted_text', 'ad_theme', 'gpt4_text_analysis'
        ]
        
        # Combine all text fields
        combined_text = []
        field_analyses = {}
        
        for field in text_fields:
            if field in row and pd.notna(row[field]):
                text = str(row[field])
                combined_text.append(text)
                field_analyses[field] = self.analyze_text(text)
        
        # Analyze combined text
        full_text = ' '.join(combined_text)
        combined_analysis = self.analyze_text(full_text)
        
        return {
            'combined_analysis': combined_analysis,
            'field_analyses': field_analyses,
            'analyzed_fields': text_fields,
            'text_length': len(full_text)
        }

    def create_feature_scores(self, analysis):
        """Create numerical scores for each feature category."""
        scores = {}
        
        for category, subcategories in analysis['combined_analysis'].items():
            category_score = 0
            subcategory_scores = {}
            
            for subcategory, data in subcategories.items():
                # Score based on number of matches and presence
                subcategory_score = min(data['count'] * 10, 100)  # Cap at 100
                subcategory_scores[subcategory] = subcategory_score
                category_score += subcategory_score
            
            scores[category] = {
                'total_score': min(category_score, 100),  # Cap at 100
                'subcategory_scores': subcategory_scores,
                'has_any_features': category_score > 0
            }
        
        return scores

    def analyze_dataset(self, df):
        """Analyze the entire dataset."""
        print("🔍 ANALYZING CAR FEATURES IN DATASET")
        print("=" * 50)
        
        results = []
        feature_summary = defaultdict(lambda: defaultdict(int))
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"   Processed {idx}/{len(df)} records...")
            
            analysis = self.analyze_ad_record(row)
            scores = self.create_feature_scores(analysis)
            
            # Create result record
            result = {
                'ad_archive_id': row.get('ad_archive_id', ''),
                'page_name': row.get('page_name', ''),
                'matched_car_models': row.get('matched_car_models', ''),
                'page_classification': row.get('page_classification', ''),
                'has_gpt4_analysis': row.get('has_gpt4_analysis', False),
                'analysis': analysis,
                'scores': scores
            }
            
            results.append(result)
            
            # Update summary statistics
            for category, data in scores.items():
                if data['has_any_features']:
                    feature_summary[category]['total_ads'] += 1
                    for subcategory, score in data['subcategory_scores'].items():
                        if score > 0:
                            feature_summary[category][f'{subcategory}_ads'] += 1
        
        print(f"✅ Analysis complete: {len(results)} records processed")
        
        return results, dict(feature_summary)


def create_features_dataframe(results):
    """Create a structured DataFrame from analysis results."""
    print("📊 Creating features DataFrame...")
    
    rows = []
    
    for result in results:
        row = {
            'ad_archive_id': result['ad_archive_id'],
            'page_name': result['page_name'],
            'matched_car_models': result['matched_car_models'],
            'page_classification': result['page_classification'],
            'has_gpt4_analysis': result['has_gpt4_analysis'],
            'text_length': result['analysis']['text_length']
        }
        
        # Add feature scores
        for category, data in result['scores'].items():
            row[f'{category}_score'] = data['total_score']
            row[f'{category}_has_features'] = data['has_any_features']
            
            for subcategory, score in data['subcategory_scores'].items():
                row[f'{category}_{subcategory}_score'] = score
        
        # Add feature matches
        for category, subcategories in result['analysis']['combined_analysis'].items():
            for subcategory, data in subcategories.items():
                row[f'{category}_{subcategory}_matches'] = '|'.join(data['matches'])
                row[f'{category}_{subcategory}_count'] = data['count']
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_summary_report(feature_summary, df_features):
    """Create a comprehensive summary report."""
    print("\n📈 FEATURE ANALYSIS SUMMARY")
    print("=" * 50)
    
    total_ads = len(df_features)
    
    print(f"📊 Total ads analyzed: {total_ads:,}")
    print(f"📊 Ads with GPT-4 analysis: {df_features['has_gpt4_analysis'].sum():,}")
    
    print(f"\n🎯 Feature Category Coverage:")
    
    for category in ['range_and_charging', 'performance', 'interior_and_comfort', 
                     'infotainment_and_audio', 'exterior_design', 'safety_and_assistance', 
                     'connectivity_and_digital']:
        
        has_features_col = f'{category}_has_features'
        if has_features_col in df_features.columns:
            count = df_features[has_features_col].sum()
            percentage = (count / total_ads) * 100
            avg_score = df_features[f'{category}_score'].mean()
            
            print(f"   {category.replace('_', ' ').title()}: {count:,} ads ({percentage:.1f}%) - Avg Score: {avg_score:.1f}")
    
    # Top models by feature coverage
    print(f"\n🚗 Top Models by Feature Coverage:")
    model_features = df_features.groupby('matched_car_models').agg({
        'range_and_charging_has_features': 'sum',
        'performance_has_features': 'sum',
        'interior_and_comfort_has_features': 'sum',
        'safety_and_assistance_has_features': 'sum'
    }).sum(axis=1).sort_values(ascending=False).head(10)
    
    for model, total_features in model_features.items():
        model_count = df_features[df_features['matched_car_models'] == model].shape[0]
        avg_features = total_features / model_count if model_count > 0 else 0
        print(f"   {model}: {total_features} total features ({avg_features:.1f} avg per ad)")


def main():
    """Main function to analyze car features."""
    print("🚗⚡ CAR FEATURES ANALYSIS")
    print("=" * 40)
    
    # Load enhanced dataset
    try:
        df = pd.read_csv('facebook_ads_electric_vehicles_enhanced_with_gpt4.csv')
        print(f"✅ Dataset loaded: {len(df)} records")
    except FileNotFoundError:
        print("❌ Enhanced dataset not found. Please run combine_gpt4_analysis.py first.")
        return
    
    # Initialize analyzer
    analyzer = CarFeaturesAnalyzer()
    
    # Analyze dataset
    results, feature_summary = analyzer.analyze_dataset(df)
    
    # Create features DataFrame
    df_features = create_features_dataframe(results)
    
    # Save results
    output_file = 'car_features_analysis.csv'
    df_features.to_csv(output_file, index=False)
    print(f"✅ Features analysis saved: {output_file}")
    
    # Save detailed results
    with open('car_features_detailed_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = []
        for result in results[:10]:  # Save first 10 for space
            json_result = {
                'ad_archive_id': result['ad_archive_id'],
                'page_name': result['page_name'],
                'matched_car_models': result['matched_car_models'],
                'scores': result['scores']
            }
            json_results.append(json_result)
        
        json.dump(json_results, f, indent=2)
    print(f"✅ Detailed results saved: car_features_detailed_results.json")
    
    # Create summary report
    create_summary_report(feature_summary, df_features)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 Results: {output_file}")
    print(f"📊 Features analyzed: 7 categories, 13 subcategories")
    print(f"🔍 Keywords matched: {sum(len(subcat) for cat in analyzer.feature_keywords.values() for subcat in cat.values())} total")


if __name__ == "__main__":
    main()
