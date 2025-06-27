python3 get_all_raw_properties.py#!/usr/bin/env python3
"""
Setup script for GPT-4 image analysis.
Helps configure OpenAI API key and test the setup.
"""

import os
import sys
import requests
import json


def check_openai_api_key(api_key: str) -> bool:
    """Test if the OpenAI API key is valid."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple test request
    test_payload = {
        "model": "gpt-4.1-nano",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            if response.status_code == 401:
                print("   Invalid API key")
            elif response.status_code == 429:
                print("   Rate limit exceeded or insufficient credits")
            else:
                print(f"   {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def setup_api_key():
    """Help user set up OpenAI API key."""
    print("🔑 OPENAI API KEY SETUP")
    print("=" * 40)
    
    # Check if already set
    existing_key = os.getenv('OPENAI_API_KEY')
    if existing_key:
        print(f"✅ API key found in environment: {'*' * (len(existing_key) - 8) + existing_key[-8:]}")
        
        # Test the key
        print("🧪 Testing API key...")
        if check_openai_api_key(existing_key):
            print("✅ API key is valid and working!")
            return existing_key
        else:
            print("❌ API key is not working properly")
    
    print("\n📝 To get your OpenAI API key:")
    print("1. Go to: https://platform.openai.com/api-keys")
    print("2. Sign in to your OpenAI account")
    print("3. Click 'Create new secret key'")
    print("4. Copy the key (starts with 'sk-')")
    
    print("\n💰 Pricing for GPT-4o-mini Vision:")
    print("- Input: $0.00015 per 1K tokens")
    print("- Output: $0.0006 per 1K tokens")
    print("- Images: ~$0.01 per image (estimated)")
    print("- 25 images ≈ $0.25")
    print("- 100 images ≈ $1.00")
    
    # Get API key from user
    try:
        api_key = input("\n🔑 Enter your OpenAI API key: ").strip()
        
        if not api_key:
            print("❌ No API key provided")
            return None
        
        if not api_key.startswith('sk-'):
            print("⚠️  Warning: API key should start with 'sk-'")
        
        # Test the key
        print("🧪 Testing API key...")
        if check_openai_api_key(api_key):
            print("✅ API key is valid!")
            
            # Ask if user wants to save it
            save_key = input("\n💾 Save API key to environment? (y/n): ").lower().strip()
            if save_key in ['y', 'yes']:
                print("\n📝 To save permanently, add this to your shell profile:")
                print(f"export OPENAI_API_KEY='{api_key}'")
                print("\nOr run:")
                print(f"echo 'export OPENAI_API_KEY=\"{api_key}\"' >> ~/.bashrc")
                print("source ~/.bashrc")
            
            return api_key
        else:
            print("❌ API key test failed")
            return None
            
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled")
        return None


def estimate_costs(num_images: int):
    """Estimate costs for image analysis."""
    print(f"\n💰 COST ESTIMATION FOR {num_images} IMAGES")
    print("=" * 40)
    
    # Rough estimates based on GPT-4o-mini pricing
    cost_per_image = 0.01  # Conservative estimate
    total_cost = num_images * cost_per_image
    
    print(f"📸 Images to analyze: {num_images}")
    print(f"💵 Estimated cost per image: ${cost_per_image:.3f}")
    print(f"💰 Total estimated cost: ${total_cost:.2f}")
    
    if total_cost > 5:
        print("⚠️  High cost warning! Consider starting with fewer images.")
    elif total_cost > 1:
        print("💡 Moderate cost. Consider testing with 10-25 images first.")
    else:
        print("✅ Low cost. Good for testing!")


def main():
    print("🚗⚡ GPT-4 IMAGE ANALYSIS SETUP")
    print("=" * 50)
    
    # Setup API key
    api_key = setup_api_key()
    
    if not api_key:
        print("\n❌ Setup failed. Cannot proceed without valid API key.")
        return
    
    # Check data file
    data_file = "facebook_ads_electric_vehicles.csv"
    if os.path.exists(data_file):
        print(f"\n✅ Data file found: {data_file}")
        
        # Quick data check
        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            image_ads = df[df['first_image_url'].notna()]
            print(f"📊 Total ads: {len(df)}")
            print(f"📸 Ads with images: {len(image_ads)}")
            
            # Cost estimation
            estimate_costs(min(len(image_ads), 25))
            estimate_costs(min(len(image_ads), 100))
            
        except Exception as e:
            print(f"⚠️  Could not analyze data file: {e}")
    else:
        print(f"\n❌ Data file not found: {data_file}")
        print("Please ensure the EV ads CSV is in the current directory.")
        return
    
    print(f"\n🚀 READY TO ANALYZE!")
    print("Run the analysis with:")
    print(f"python3 gpt4_image_analyzer.py {api_key} 25")
    print("\nOr set the environment variable and run:")
    print(f"export OPENAI_API_KEY='{api_key}'")
    print("python3 gpt4_image_analyzer.py")
    
    # Ask if user wants to run now
    try:
        run_now = input("\n▶️  Run analysis now with 10 test images? (y/n): ").lower().strip()
        if run_now in ['y', 'yes']:
            print("\n🔄 Starting analysis...")
            os.system(f"python3 gpt4_image_analyzer.py {api_key} 10")
    except KeyboardInterrupt:
        print("\n👋 Setup complete!")


if __name__ == "__main__":
    main()
