#!/usr/bin/env python3
"""
Setup script for the Ecosystem Map Generator project.
This script helps install dependencies and set up the environment.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False
    return True

def check_chromedriver():
    """Check if Chrome and chromedriver are available"""
    try:
        subprocess.run(["chromedriver", "--version"], capture_output=True, check=True)
        print("✓ ChromeDriver is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ ChromeDriver not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "webdriver-manager"])
            print("✓ WebDriver Manager installed (will auto-download ChromeDriver)")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install WebDriver Manager")
            return False

def setup_environment():
    """Set up the environment file"""
    env_file = ".env"
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            f.write("# OpenAI API Key (required)\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("\n# Optional: Chrome binary path (if not in PATH)\n")
            f.write("# CHROME_BINARY_PATH=/path/to/chrome\n")
        print(f"✓ Created {env_file} template")
        print("  Please edit .env and add your OpenAI API key")
    else:
        print(f"✓ {env_file} already exists")

def create_sample_data():
    """Create sample data without pandas dependency"""
    companies = [
        "Benchling",
        "Ginkgo Bioworks", 
        "10x Genomics",
        "Illumina",
        "Twist Bioscience",
        "Berkeley Lights",
        "Zymergen",
        "Synthetic Genomics",
        "Amyris",
        "Genomatica",
        "Moderna",
        "BioNTech",
        "Recursion Pharmaceuticals",
        "Atomwise",
        "Insitro",
        "Schrödinger",
        "CureVac",
        "AbCellera",
        "Generate Biomedicines",
        "DeepMind"
    ]
    
    # Create simple CSV first, then convert to Excel after pandas is installed
    with open('companies.csv', 'w') as f:
        f.write('CompanyName\n')
        for company in companies:
            f.write(f'{company}\n')
    
    print(f"✓ Created companies.csv with {len(companies)} companies")

def main():
    """Main setup function"""
    print("=== Ecosystem Map Generator Setup ===\n")
    
    # Step 1: Install requirements
    if not install_requirements():
        print("Setup failed. Please resolve the above errors.")
        return
    
    # Step 2: Check ChromeDriver
    check_chromedriver()
    
    # Step 3: Set up environment
    setup_environment()
    
    # Step 4: Create sample data
    create_sample_data()
    
    # Step 5: Convert CSV to Excel now that pandas is available
    try:
        import pandas as pd
        df = pd.read_csv('companies.csv')
        df.to_excel('companies.xlsx', index=False)
        os.remove('companies.csv')  # Clean up CSV
        print("✓ Created companies.xlsx")
    except ImportError:
        print("⚠ Could not create Excel file yet. Run create_sample_data.py after installing pandas.")
    
    print("\n=== Setup Complete! ===")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: python3 ecosystem_map.py")
    print("\nNote: The script requires Chrome browser to be installed for web scraping.")

if __name__ == "__main__":
    main()
