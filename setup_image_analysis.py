#!/usr/bin/env python3
"""
Setup script for image analysis dependencies.
Installs required packages for AI-powered image classification.
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
    print("=== SETTING UP IMAGE ANALYSIS ENVIRONMENT ===")
    
    # Required packages
    packages = [
        "torch",
        "transformers", 
        "pillow",
        "opencv-python",
        "numpy",
        "requests"
    ]
    
    # Check what's already installed
    print("\nChecking existing packages...")
    installed = []
    missing = []
    
    for package in packages:
        # Map package names to import names
        import_name = package
        if package == "opencv-python":
            import_name = "cv2"
        elif package == "pillow":
            import_name = "PIL"
        
        if check_package(import_name):
            print(f"✓ {package} already installed")
            installed.append(package)
        else:
            print(f"✗ {package} not found")
            missing.append(package)
    
    if not missing:
        print("\n🎉 All packages already installed!")
        return
    
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
    
    if success_count == len(missing):
        print("🎉 All packages installed successfully!")
        print("\nYou can now run image analysis with:")
        print("python3 classify_ad_images.py")
    else:
        print("⚠️  Some packages failed to install.")
        print("You may need to install them manually:")
        for package in missing:
            print(f"  pip install {package}")
    
    # Test the installation
    print("\n=== TESTING INSTALLATION ===")
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        print("✗ PyTorch not available")
    
    try:
        from transformers import BlipProcessor
        print("✓ Transformers (BLIP) available")
    except ImportError:
        print("✗ Transformers not available")
    
    try:
        from transformers import CLIPProcessor
        print("✓ CLIP model available")
    except ImportError:
        print("✗ CLIP model not available")
    
    try:
        import cv2
        print("✓ OpenCV available")
    except ImportError:
        print("✗ OpenCV not available (optional)")
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow available")
    except ImportError:
        print("✗ PIL/Pillow not available")


if __name__ == "__main__":
    main()
