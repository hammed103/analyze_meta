#!/usr/bin/env python3
"""
Quick launcher for the EV Ads Dashboard with Theme Analysis
Optimized to display pre-computed theme analysis results only
"""

import subprocess
import sys
import os

def check_theme_files():
    """Check if theme analysis files exist"""
    theme_files = [
        'theme_frequency_overall.csv',
        'theme_frequency_by_model.csv', 
        'lightweight_nlp_keywords.csv',
        'lightweight_nlp_model_themes.csv',
        'lightweight_nlp_themes.csv'
    ]
    
    existing_files = []
    missing_files = []
    
    for file in theme_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    return existing_files, missing_files

def main():
    print("🎨 EV Ads Dashboard - Theme Analysis Launcher")
    print("=" * 50)
    
    # Check theme files
    existing, missing = check_theme_files()
    
    print(f"\n📊 Theme Analysis Files Status:")
    print(f"✅ Found: {len(existing)} files")
    for file in existing:
        print(f"   • {file}")
    
    if missing:
        print(f"\n⚠️  Missing: {len(missing)} files")
        for file in missing:
            print(f"   • {file}")
        
        print(f"\n💡 To generate missing files:")
        print(f"   python3 simple_theme_analysis.py")
        print(f"   python3 lightweight_nlp_analysis.py")
    
    print(f"\n🚀 Launching Dashboard...")
    print(f"📍 Theme Analysis will be available in the '🎨 Theme Analysis' tab")
    print(f"🔗 Dashboard will open at: http://localhost:8501")
    
    # Launch dashboard
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "ev_ads_dashboard.py",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print(f"\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print(f"\n🔧 Try running manually:")
        print(f"   streamlit run ev_ads_dashboard.py")

if __name__ == "__main__":
    main()
