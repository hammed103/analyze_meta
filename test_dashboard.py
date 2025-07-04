#!/usr/bin/env python3
"""
Test script to verify dashboard functionality
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
        
        import pandas as pd
        print("✓ Pandas imported successfully")
        
        import plotly.express as px
        print("✓ Plotly imported successfully")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        print("✓ Scikit-learn imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_files():
    """Test if required data files exist"""
    required_files = [
        'Data/facebook_ads_electric_vehicles_with_openai_summaries_cached.csv'
    ]
    
    optional_files = [
        'theme_frequency_overall.csv',
        'theme_frequency_by_model.csv',
        'lightweight_nlp_keywords.csv',
        'lightweight_nlp_model_themes.csv',
        'lightweight_nlp_themes.csv'
    ]
    
    print("\nChecking required data files:")
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
    
    print("\nChecking optional theme analysis files:")
    for file in optional_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"- {file} not found (will use live analysis)")

def test_dashboard_syntax():
    """Test dashboard syntax"""
    try:
        import py_compile
        py_compile.compile('ev_ads_dashboard.py', doraise=True)
        print("✓ Dashboard syntax is valid")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ Dashboard syntax error: {e}")
        return False

def main():
    print("🧪 Testing Dashboard Setup")
    print("=" * 40)
    
    # Test imports
    print("\n1. Testing imports...")
    imports_ok = test_imports()
    
    # Test data files
    print("\n2. Testing data files...")
    test_data_files()
    
    # Test syntax
    print("\n3. Testing dashboard syntax...")
    syntax_ok = test_dashboard_syntax()
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    
    if imports_ok and syntax_ok:
        print("✅ Dashboard should work correctly!")
        print("\n🚀 To run the dashboard:")
        print("   streamlit run ev_ads_dashboard.py")
        print("\n📝 Note: If theme analysis files are missing,")
        print("   the dashboard will perform live analysis on filtered data.")
    else:
        print("❌ Dashboard has issues that need to be resolved.")
        
        if not imports_ok:
            print("   - Fix import issues first")
        if not syntax_ok:
            print("   - Fix syntax errors in ev_ads_dashboard.py")

if __name__ == "__main__":
    main()
