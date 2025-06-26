#!/usr/bin/env python3
"""
Launch script for the EV Ads Dashboard
"""

import subprocess
import sys
import os


def main():
    print("🚗⚡ Launching EV Ads Analysis Dashboard...")
    
    # Check if data file exists
    data_file = "facebook_ads_electric_vehicles.csv"
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file '{data_file}' not found!")
        print("Please ensure the EV ads CSV file is in the current directory.")
        return
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✓ Streamlit found")
    except ImportError:
        print("❌ Streamlit not installed. Please run:")
        print("  python3 setup_dashboard.py")
        return
    
    # Launch the dashboard
    try:
        print("🚀 Starting dashboard on http://localhost:8501")
        print("Press Ctrl+C to stop the dashboard")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ev_ads_dashboard.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")


if __name__ == "__main__":
    main()
