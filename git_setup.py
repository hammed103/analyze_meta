#!/usr/bin/env python3
"""
Git setup script for Meta Analysis project.
Initializes git repository and checks what files will be committed.
"""

import subprocess
import os
import sys


def run_command(command, description):
    """Run a shell command and return the result."""
    try:
        print(f"\n🔄 {description}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} successful")
            if result.stdout.strip():
                print(result.stdout)
        else:
            print(f"❌ {description} failed")
            if result.stderr.strip():
                print(f"Error: {result.stderr}")
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def check_git_status():
    """Check current git status and what files would be committed."""
    print("\n📋 CHECKING GIT STATUS")
    print("=" * 50)
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("📁 Git repository not initialized")
        return False
    
    # Show git status
    run_command("git status", "Checking git status")
    
    # Show what files are tracked/untracked
    print("\n📄 FILES THAT WILL BE COMMITTED:")
    run_command("git ls-files", "Listing tracked files")
    
    print("\n🚫 FILES THAT ARE IGNORED:")
    run_command("git status --ignored", "Checking ignored files")
    
    return True


def initialize_git():
    """Initialize git repository and add files."""
    print("\n🚀 INITIALIZING GIT REPOSITORY")
    print("=" * 50)
    
    # Initialize git
    if not run_command("git init", "Initializing git repository"):
        return False
    
    # Add all files (respecting .gitignore)
    if not run_command("git add .", "Adding files to git"):
        return False
    
    # Show what will be committed
    print("\n📋 FILES TO BE COMMITTED:")
    run_command("git status --cached", "Checking staged files")
    
    return True


def verify_no_data_files():
    """Verify that no data files are being tracked."""
    print("\n🔍 VERIFYING NO DATA FILES ARE TRACKED")
    print("=" * 50)
    
    # Check for CSV files
    result = subprocess.run("git ls-files | grep -E '\\.(csv|json)$'", 
                          shell=True, capture_output=True, text=True)
    
    if result.stdout.strip():
        print("⚠️  WARNING: Data files found in git:")
        print(result.stdout)
        print("\n❌ PLEASE REMOVE THESE FILES FROM GIT:")
        print("git rm --cached <filename>")
        return False
    else:
        print("✅ No data files (.csv, .json) are being tracked")
        return True


def main():
    print("🚗⚡ META ANALYSIS - GIT SETUP")
    print("=" * 50)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check if .gitignore exists
    if os.path.exists('.gitignore'):
        print("✅ .gitignore file found")
    else:
        print("❌ .gitignore file not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Check if git is already initialized
    if os.path.exists('.git'):
        print("📁 Git repository already exists")
        check_git_status()
    else:
        print("📁 No git repository found")
        
        # Ask user if they want to initialize
        try:
            response = input("\nWould you like to initialize a git repository? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                if initialize_git():
                    print("\n✅ Git repository initialized successfully!")
                else:
                    print("\n❌ Failed to initialize git repository")
                    return
            else:
                print("👋 Git initialization skipped")
                return
        except KeyboardInterrupt:
            print("\n👋 Setup cancelled")
            return
    
    # Verify no data files are tracked
    if not verify_no_data_files():
        print("\n⚠️  Please fix the data file issues before proceeding")
        return
    
    print("\n🎉 GIT SETUP COMPLETE!")
    print("\nNext steps:")
    print("1. Review the files that will be committed:")
    print("   git status")
    print("2. Make your first commit:")
    print("   git commit -m 'Initial commit: Meta Analysis EV Ads Dashboard'")
    print("3. Add remote repository:")
    print("   git remote add origin <your-github-repo-url>")
    print("4. Push to GitHub:")
    print("   git push -u origin main")
    
    print("\n🔒 IMPORTANT:")
    print("- No data files (.csv, .json) will be committed")
    print("- Only analysis code and documentation will be pushed")
    print("- Your data remains private and local")


if __name__ == "__main__":
    main()
