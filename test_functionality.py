#!/usr/bin/env python3
"""
Test script to validate key functionality of the ecosystem map generator.
This script tests individual components without running the full pipeline.
"""

import pandas as pd
import os
import logging
from ecosystem_map import EcosystemMapGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_excel_loading():
    """Test loading companies from Excel file"""
    print("=== Testing Excel Loading ===")
    try:
        df = pd.read_excel('companies.xlsx')
        print(f"✓ Loaded {len(df)} companies from Excel")
        print("Sample companies:")
        for i, company in enumerate(df['CompanyName'].head(5)):
            print(f"  {i+1}. {company}")
        return True
    except Exception as e:
        print(f"✗ Failed to load Excel: {e}")
        return False

def test_environment_setup():
    """Test environment variables and configuration"""
    print("\n=== Testing Environment Setup ===")
    
    # Check for .env file
    if os.path.exists('.env'):
        print("✓ .env file found")
    else:
        print("✗ .env file missing")
        return False
    
    # Check OpenAI API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != 'your_openai_api_key_here':
        print("✓ OpenAI API key configured")
        return True
    else:
        print("⚠ OpenAI API key not configured (edit .env file)")
        return False

def test_selenium_setup():
    """Test Selenium WebDriver setup"""
    print("\n=== Testing Selenium Setup ===")
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Test ChromeDriver installation
        driver_path = ChromeDriverManager().install()
        print(f"✓ ChromeDriver available at: {driver_path}")
        
        # Test WebDriver initialization
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://www.google.com")
        title = driver.title
        driver.quit()
        
        print(f"✓ WebDriver test successful (loaded: {title})")
        return True
        
    except Exception as e:
        print(f"✗ Selenium setup failed: {e}")
        return False

def test_data_processing():
    """Test data processing functions"""
    print("\n=== Testing Data Processing ===")
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Test data normalization
        sample_data = [10, 50, 100, 25, 75]
        normalized = [(x - min(sample_data)) / (max(sample_data) - min(sample_data)) * 100 
                     for x in sample_data]
        
        print(f"✓ Sample data: {sample_data}")
        print(f"✓ Normalized: {[round(x, 1) for x in normalized]}")
        
        # Test plot creation (don't show, just test)
        plt.figure(figsize=(8, 6))
        plt.scatter([1, 2, 3], [1, 2, 3])
        plt.title("Test Plot")
        plt.close()  # Close without showing
        
        print("✓ Plotting functionality working")
        return True
        
    except Exception as e:
        print(f"✗ Data processing test failed: {e}")
        return False

def test_sample_functionality():
    """Test a sample of the main functionality without full execution"""
    print("\n=== Testing Sample Functionality ===")
    
    # Test text extraction function
    from ecosystem_map import EcosystemMapGenerator
    
    # Mock generator to test number extraction
    class MockGenerator:
        def _extract_number(self, text, context):
            gen = EcosystemMapGenerator("dummy_key")  # Won't be used for this test
            return gen._extract_number(text, context)
    
    mock = MockGenerator()
    
    # Test number extraction
    test_cases = [
        ("The company raised $50 million in funding", "funding", 50.0),
        ("Revenue of 2.5 billion dollars", "revenue", 2500.0),
        ("They have 150 employees", "employees", 0.00015),  # Converted to millions
        ("No financial data available", "funding", 0.0)
    ]
    
    all_passed = True
    for text, context, expected in test_cases:
        result = mock._extract_number(text, context)
        if abs(result - expected) < 0.001:  # Allow small floating point differences
            print(f"✓ Number extraction test passed: '{text[:30]}...' -> {result}")
        else:
            print(f"✗ Number extraction test failed: expected {expected}, got {result}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("Ecosystem Map Generator - Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_excel_loading,
        test_environment_setup,
        test_data_processing,
        test_sample_functionality,
        test_selenium_setup,  # This one last as it might take longer
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo run the full analysis:")
        print("1. Edit .env file and add your OpenAI API key")
        print("2. Run: python3 ecosystem_map.py")
    else:
        print("⚠ Some tests failed. Please resolve the issues above.")
        
        if not results[1]:  # Environment setup failed
            print("\nCritical: OpenAI API key must be configured to run the full analysis.")

if __name__ == "__main__":
    main()
