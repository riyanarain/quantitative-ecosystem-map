#!/usr/bin/env python3
"""
Quantitative Ecosystem Map Generator

This script generates a 2D scatter plot visualization of companies based on:
- X-axis: Dominance metrics (financial strength, market penetration)
- Y-axis: AI Ready metrics (API quality, platform architecture, native AI features)

The script automates data collection using web scraping and LLM reasoning,
calculates normalized scores, and outputs a labeled scatter plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from urllib.parse import quote

# Third-party imports (to be installed)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MetricWeights:
    """Configuration for metric weights"""
    x_axis = {
        'FinancialStrength': 0.4,
        'MarketPenetration': 0.6,
    }
    y_axis = {
        'APIScore': 0.5,
        'PlatformArchitecture': 0.3,
        'NativeAIFeatures': 0.2,
    }

class EcosystemMapGenerator:
    """Main class for generating ecosystem maps"""
    
    def __init__(self, openai_api_key: str, input_file: str = "companies.xlsx"):
        """
        Initialize the ecosystem map generator
        
        Args:
            openai_api_key: OpenAI API key for LLM analysis
            input_file: Path to Excel file containing company names
        """
        self.openai_api_key = openai_api_key
        self.input_file = input_file
        self.df = None
        self.driver = None
        self.weights = MetricWeights()
        
        # Initialize OpenAI client
        openai.api_key = openai_api_key
        
        # Setup Selenium driver
        self._setup_selenium()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver with appropriate options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def load_companies(self) -> pd.DataFrame:
        """
        Load company names from Excel file
        
        Returns:
            DataFrame with company names
        """
        try:
            self.df = pd.read_excel(self.input_file)
            if 'CompanyName' not in self.df.columns:
                raise ValueError("Excel file must contain a 'CompanyName' column")
            
            logger.info(f"Loaded {len(self.df)} companies from {self.input_file}")
            return self.df
        except Exception as e:
            logger.error(f"Failed to load companies: {e}")
            raise
    
    def search_perplexity(self, query: str, max_retries: int = 3) -> str:
        """
        Perform a search on Perplexity AI and extract the result
        
        Args:
            query: Search query string
            max_retries: Maximum number of retry attempts
            
        Returns:
            Extracted search result text
        """
        for attempt in range(max_retries):
            try:
                # Navigate to Perplexity
                self.driver.get("https://www.perplexity.ai/")
                
                # Wait for and find the search input
                search_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "textarea, input[type='text']"))
                )
                
                # Clear and enter the query
                search_input.clear()
                search_input.send_keys(query)
                
                # Submit the search (look for submit button or press Enter)
                try:
                    submit_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit'], button[aria-label*='Search']")
                    submit_button.click()
                except NoSuchElementException:
                    # If no submit button, try pressing Enter
                    from selenium.webdriver.common.keys import Keys
                    search_input.send_keys(Keys.RETURN)
                
                # Wait for results to load
                time.sleep(5)
                
                # Extract the response text
                response_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid*='response'], .response, .answer")
                
                if response_elements:
                    result = response_elements[0].text.strip()
                    logger.info(f"Successfully extracted result for query: {query[:50]}...")
                    return result
                else:
                    # Fallback: get page text
                    page_text = self.driver.find_element(By.TAG_NAME, "body").text
                    # Extract relevant portion (first 500 chars after the query)
                    result = page_text[:1000].strip()
                    return result
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for query '{query}': {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"All attempts failed for query: {query}")
                    return ""
        
        return ""
    
    def extract_financial_metrics(self, company_name: str) -> Dict[str, float]:
        """
        Extract financial strength metrics for a company
        
        Args:
            company_name: Name of the company
            
        Returns:
            Dictionary with financial metrics
        """
        metrics = {}
        
        # Query for total funding
        funding_query = f"Total funding raised by {company_name} company in millions USD"
        funding_result = self.search_perplexity(funding_query)
        metrics['TotalFunding'] = self._extract_number(funding_result, 'funding')
        
        # Query for market cap (for public companies)
        market_cap_query = f"Market capitalization of {company_name} in millions USD"
        market_cap_result = self.search_perplexity(market_cap_query)
        metrics['MarketCap'] = self._extract_number(market_cap_result, 'market cap')
        
        # Query for annual revenue
        revenue_query = f"Annual revenue of {company_name} in millions USD latest year"
        revenue_result = self.search_perplexity(revenue_query)
        metrics['Revenue'] = self._extract_number(revenue_result, 'revenue')
        
        # Calculate financial strength (use the highest available metric)
        financial_values = [v for v in [metrics['TotalFunding'], metrics['MarketCap'], metrics['Revenue']] if v > 0]
        metrics['FinancialStrength'] = max(financial_values) if financial_values else 0
        
        logger.info(f"Financial metrics for {company_name}: {metrics}")
        return metrics
    
    def extract_market_metrics(self, company_name: str) -> Dict[str, float]:
        """
        Extract market penetration metrics for a company
        
        Args:
            company_name: Name of the company
            
        Returns:
            Dictionary with market metrics
        """
        metrics = {}
        
        # Query for employee count
        employee_query = f"Number of employees at {company_name} company current total"
        employee_result = self.search_perplexity(employee_query)
        metrics['EmployeeCount'] = self._extract_number(employee_result, 'employees')
        
        # Query for web traffic
        traffic_query = f"Monthly web traffic visitors for {company_name} website"
        traffic_result = self.search_perplexity(traffic_query)
        metrics['WebTraffic'] = self._extract_number(traffic_result, 'traffic')
        
        # Calculate market penetration score
        employee_score = min(metrics['EmployeeCount'] / 10, 100) if metrics['EmployeeCount'] > 0 else 0
        traffic_score = min(metrics['WebTraffic'] / 100000, 100) if metrics['WebTraffic'] > 0 else 0
        metrics['MarketPenetration'] = max(employee_score, traffic_score)
        
        logger.info(f"Market metrics for {company_name}: {metrics}")
        return metrics
    
    def _extract_number(self, text: str, context: str) -> float:
        """
        Extract numerical values from text using regex
        
        Args:
            text: Text to extract numbers from
            context: Context for logging (e.g., 'funding', 'employees')
            
        Returns:
            Extracted number or 0 if not found
        """
        if not text:
            return 0.0
        
        # Remove common text and normalize
        text = text.lower().replace(',', '').replace('$', '')
        
        # Pattern to match numbers with units (million, billion, thousand, etc.)
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:billion|b)\b',  # Billions
            r'(\d+(?:\.\d+)?)\s*(?:million|m)\b',  # Millions  
            r'(\d+(?:\.\d+)?)\s*(?:thousand|k)\b', # Thousands
            r'(\d+(?:\.\d+)?)',  # Raw numbers
        ]
        
        # Different multipliers based on context
        if context in ['funding', 'market cap', 'revenue']:
            # Financial metrics: convert to millions
            multipliers = [1000, 1, 0.001, 0.000001]
        elif context in ['employees']:
            # Employee count: keep as actual number
            multipliers = [1000000000, 1000000, 1000, 1]
        elif context in ['traffic']:
            # Web traffic: keep as actual number
            multipliers = [1000000000, 1000000, 1000, 1]
        else:
            # Default: convert to millions
            multipliers = [1000, 1, 0.001, 0.000001]
        
        for pattern, multiplier in zip(patterns, multipliers):
            matches = re.findall(pattern, text)
            if matches:
                try:
                    number = float(matches[0]) * multiplier
                    logger.debug(f"Extracted {context}: {number} from text: {text[:100]}...")
                    return number
                except ValueError:
                    continue
        
        logger.warning(f"Could not extract {context} from text: {text[:100]}...")
        return 0.0
    
    def analyze_api_quality(self, company_name: str) -> int:
        """
        Analyze API quality using OpenAI
        
        Args:
            company_name: Name of the company
            
        Returns:
            API quality score (0-3)
        """
        prompt = f"""Analyze the public documentation for {company_name} to assess its API quality. 
        Provide a single integer score from 0 to 3 based on this rubric:
        0: No public API found.
        1: API exists but is limited or poorly documented.
        2: A comprehensive, well-documented REST API is available.
        3: A comprehensive API plus a dedicated Python SDK/toolkit for data scientists is available.
        
        Return only the integer score."""
        
        return self._call_openai_for_score(prompt, company_name, "API Quality")
    
    def analyze_platform_architecture(self, company_name: str) -> int:
        """
        Analyze platform architecture using OpenAI
        
        Args:
            company_name: Name of the company
            
        Returns:
            Platform architecture score (1-3)
        """
        prompt = f"""Based on the product descriptions for {company_name}, classify its platform architecture. 
        Provide a single integer score from 1 to 3 based on this rubric:
        1: Legacy on-premise software.
        2: 'Cloud-enabled' or a hosted version of legacy software.
        3: Modern, multi-tenant, cloud-native SaaS platform.
        
        Return only the integer score."""
        
        return self._call_openai_for_score(prompt, company_name, "Platform Architecture")
    
    def analyze_ai_features(self, company_name: str) -> int:
        """
        Analyze native AI/ML features using OpenAI
        
        Args:
            company_name: Name of the company
            
        Returns:
            AI features score (0 or 1)
        """
        prompt = f"""Review the product features of {company_name}. Does the platform offer native AI or machine learning capabilities? 
        Provide a single integer score of 0 (No) or 1 (Yes). 
        
        Return only the integer score."""
        
        return self._call_openai_for_score(prompt, company_name, "AI Features")
    
    def _call_openai_for_score(self, prompt: str, company_name: str, metric_name: str, max_retries: int = 3) -> int:
        """
        Call OpenAI API to get a score
        
        Args:
            prompt: The prompt to send to OpenAI
            company_name: Company name for logging
            metric_name: Name of the metric for logging
            max_retries: Maximum retry attempts
            
        Returns:
            Integer score from the LLM
        """
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0
                )
                
                result = response.choices[0].message.content.strip()
                
                # Extract integer from response
                score_match = re.search(r'\b(\d+)\b', result)
                if score_match:
                    score = int(score_match.group(1))
                    logger.info(f"{metric_name} score for {company_name}: {score}")
                    return score
                else:
                    logger.warning(f"Could not extract score from OpenAI response: {result}")
                    
            except Exception as e:
                logger.error(f"OpenAI API call failed for {company_name} ({metric_name}), attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        logger.error(f"All OpenAI attempts failed for {company_name} ({metric_name})")
        return 0
    
    def process_all_companies(self):
        """Process all companies to collect metrics"""
        if self.df is None:
            raise ValueError("No companies loaded. Call load_companies() first.")
        
        total_companies = len(self.df)
        
        for idx, row in self.df.iterrows():
            company_name = row['CompanyName']
            logger.info(f"Processing company {idx + 1}/{total_companies}: {company_name}")
            
            # Extract quantitative metrics
            financial_metrics = self.extract_financial_metrics(company_name)
            market_metrics = self.extract_market_metrics(company_name)
            
            # Add to dataframe
            for key, value in {**financial_metrics, **market_metrics}.items():
                self.df.at[idx, key] = value
            
            # Extract qualitative metrics using OpenAI
            self.df.at[idx, 'APIScore'] = self.analyze_api_quality(company_name)
            time.sleep(1)  # Rate limiting
            
            self.df.at[idx, 'PlatformArchitecture'] = self.analyze_platform_architecture(company_name)
            time.sleep(1)  # Rate limiting
            
            self.df.at[idx, 'NativeAIFeatures'] = self.analyze_ai_features(company_name)
            time.sleep(1)  # Rate limiting
            
            # Save progress periodically
            if (idx + 1) % 5 == 0:
                self.df.to_excel(f"ecosystem_data_progress_{idx + 1}.xlsx", index=False)
                logger.info(f"Saved progress after {idx + 1} companies")
    
    def normalize_data(self):
        """Normalize all metric columns to 0-100 scale"""
        if self.df is None:
            raise ValueError("No data to normalize")
        
        # Columns to normalize
        metric_columns = ['FinancialStrength', 'MarketPenetration', 'APIScore', 'PlatformArchitecture', 'NativeAIFeatures']
        
        for col in metric_columns:
            if col in self.df.columns:
                # Min-max normalization to 0-100 scale
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                
                if col_max > col_min:
                    self.df[f'{col}_Normalized'] = 100 * (self.df[col] - col_min) / (col_max - col_min)
                else:
                    self.df[f'{col}_Normalized'] = 50  # If all values are the same
                    
                logger.info(f"Normalized {col}: min={col_min}, max={col_max}")
    
    def calculate_final_scores(self):
        """Calculate weighted final scores for X and Y axes"""
        if self.df is None:
            raise ValueError("No data to calculate scores")
        
        # Calculate X-axis score (Dominance)
        x_components = []
        for metric, weight in self.weights.x_axis.items():
            col_name = f'{metric}_Normalized'
            if col_name in self.df.columns:
                x_components.append(self.df[col_name] * weight)
            else:
                logger.warning(f"Missing normalized column: {col_name}")
        
        if x_components:
            self.df['Dominance_Score'] = sum(x_components)
        else:
            self.df['Dominance_Score'] = 0
        
        # Calculate Y-axis score (AI Ready)
        y_components = []
        for metric, weight in self.weights.y_axis.items():
            col_name = f'{metric}_Normalized'
            if col_name in self.df.columns:
                y_components.append(self.df[col_name] * weight)
            else:
                logger.warning(f"Missing normalized column: {col_name}")
        
        if y_components:
            self.df['AI_Ready_Score'] = sum(y_components)
        else:
            self.df['AI_Ready_Score'] = 0
        
        logger.info("Final scores calculated successfully")
    
    def create_visualization(self, output_file: str = "ecosystem_map.png"):
        """
        Create and save the ecosystem map visualization
        
        Args:
            output_file: Output filename for the plot
        """
        if self.df is None:
            raise ValueError("No data to visualize")
        
        # Set up the plot style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            # Fallback if seaborn style is not available
            plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot
        scatter = ax.scatter(
            self.df['Dominance_Score'],
            self.df['AI_Ready_Score'],
            s=100,
            alpha=0.7,
            c='steelblue',
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add company labels
        for idx, row in self.df.iterrows():
            ax.annotate(
                row['CompanyName'],
                (row['Dominance_Score'], row['AI_Ready_Score']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                ha='left',
                va='bottom'
            )
        
        # Customize the plot
        ax.set_xlabel('Dominance in the Ecosystem', fontsize=14, fontweight='bold')
        ax.set_ylabel('AI Ready', fontsize=14, fontweight='bold')
        ax.set_title('Quantitative Ecosystem Map', fontsize=16, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set axis limits with some padding
        x_margin = (self.df['Dominance_Score'].max() - self.df['Dominance_Score'].min()) * 0.1
        y_margin = (self.df['AI_Ready_Score'].max() - self.df['AI_Ready_Score'].min()) * 0.1
        
        ax.set_xlim(
            self.df['Dominance_Score'].min() - x_margin,
            self.df['Dominance_Score'].max() + x_margin
        )
        ax.set_ylim(
            self.df['AI_Ready_Score'].min() - y_margin,
            self.df['AI_Ready_Score'].max() + y_margin
        )
        
        # Tight layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Ecosystem map saved as {output_file}")
    
    def save_results(self, output_file: str = "ecosystem_data_final.xlsx"):
        """
        Save the final results to Excel
        
        Args:
            output_file: Output filename for the Excel file
        """
        if self.df is None:
            raise ValueError("No data to save")
        
        self.df.to_excel(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")
    
    def run_full_pipeline(self, output_plot: str = "ecosystem_map.png", output_data: str = "ecosystem_data_final.xlsx"):
        """
        Run the complete ecosystem mapping pipeline
        
        Args:
            output_plot: Filename for the output plot
            output_data: Filename for the output data
        """
        try:
            logger.info("Starting ecosystem map generation pipeline")
            
            # Step 1: Load companies
            self.load_companies()
            
            # Step 2: Process all companies
            self.process_all_companies()
            
            # Step 3: Normalize data
            self.normalize_data()
            
            # Step 4: Calculate final scores
            self.calculate_final_scores()
            
            # Step 5: Create visualization
            self.create_visualization(output_plot)
            
            # Step 6: Save results
            self.save_results(output_data)
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.cleanup()


def main():
    """Main function to run the ecosystem map generator"""
    
    # Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    INPUT_FILE = "companies.xlsx"
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found. Please create it with a 'CompanyName' column.")
        return
    
    # Initialize and run the generator
    generator = EcosystemMapGenerator(OPENAI_API_KEY, INPUT_FILE)
    generator.run_full_pipeline()


if __name__ == "__main__":
    main()
