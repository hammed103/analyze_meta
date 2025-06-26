#!/usr/bin/env python3
"""
Setup script for the EV Ads Dashboard
Installs required packages and provides launch instructions
"""

import subprocess
import sys
import os


def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False


def check_package(package_name):
    """Check if a package is already installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def main():
    print("=== EV ADS DASHBOARD SETUP ===")
    
    # Required packages for the dashboard
    packages = [
        "streamlit",
        "plotly", 
        "pandas",
        "numpy",
        "pillow",
        "requests"
    ]
    
    # Check what's already installed
    print("\nChecking existing packages...")
    installed = []
    missing = []
    
    for package in packages:
        if check_package(package):
            print(f"✓ {package} already installed")
            installed.append(package)
        else:
            print(f"✗ {package} not found")
            missing.append(package)
    
    if not missing:
        print("\n🎉 All packages already installed!")
    else:
        print(f"\nInstalling {len(missing)} missing packages...")
        
        # Install missing packages
        success_count = 0
        for package in missing:
            print(f"\nInstalling {package}...")
            if install_package(package):
                print(f"✓ Successfully installed {package}")
                success_count += 1
            else:
                print(f"✗ Failed to install {package}")
        
        print(f"\n=== INSTALLATION SUMMARY ===")
        print(f"Successfully installed: {success_count}/{len(missing)} packages")
        
        if success_count < len(missing):
            print("⚠️  Some packages failed to install.")
            print("You may need to install them manually:")
            for package in missing:
                print(f"  pip install {package}")
            return
    
    # Check if data file exists
    data_file = "facebook_ads_electric_vehicles.csv"
    if os.path.exists(data_file):
        print(f"\n✓ Data file found: {data_file}")
    else:
        print(f"\n⚠️  Data file not found: {data_file}")
        print("Please ensure the EV ads CSV file is in the current directory.")
    
    # Test streamlit installation
    print("\n=== TESTING STREAMLIT ===")
    try:
        import streamlit as st
        print("✓ Streamlit successfully imported")
    except ImportError:
        print("✗ Streamlit import failed")
        return
    
    print("\n🎉 Setup complete!")
    print("\n=== LAUNCH INSTRUCTIONS ===")
    print("To start the dashboard, run:")
    print("  streamlit run ev_ads_dashboard.py")
    print("\nOr use the launch script:")
    print("  python3 launch_dashboard.py")
    
    # Ask if user wants to launch now
    try:
        launch_now = input("\nWould you like to launch the dashboard now? (y/n): ").lower().strip()
        if launch_now in ['y', 'yes']:
            print("\nLaunching dashboard...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "ev_ads_dashboard.py"])
    except KeyboardInterrupt:
        print("\nSetup completed. You can launch the dashboard later with:")
        print("  streamlit run ev_ads_dashboard.py")


if __name__ == "__main__":
    main()
