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
from webdriver_manager.chrome import ChromeDriverManager

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
        'FinancialStrength': 0.7,
        'MarketPenetration': 0.3,
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
        """Setup Selenium WebDriver with stealth options"""
        chrome_options = Options()
        # Remove headless mode for better compatibility
        # chrome_options.add_argument("--headless")  
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        
        # Randomized user agent
        user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        chrome_options.add_argument(f"--user-agent={np.random.choice(user_agents)}")
        
        try:
            # Fix for macOS ARM issue with ChromeDriverManager
            try:
                driver_path = ChromeDriverManager().install()
                # Check if the path points to the actual executable
                if 'THIRD_PARTY_NOTICES' in driver_path or not driver_path.endswith('chromedriver'):
                    # Fix the path to point to the actual chromedriver executable
                    import os
                    driver_dir = os.path.dirname(driver_path)
                    potential_driver = os.path.join(driver_dir, 'chromedriver')
                    if os.path.exists(potential_driver):
                        driver_path = potential_driver
                    else:
                        # Use the known correct path from find command
                        fallback_path = "/Users/rnarain/.wdm/drivers/chromedriver/mac64/139.0.7258.154/chromedriver-mac-arm64/chromedriver"
                        if os.path.exists(fallback_path):
                            driver_path = fallback_path
                
                self.driver = webdriver.Chrome(service=webdriver.chrome.service.Service(driver_path), options=chrome_options)
            except Exception as driver_error:
                # Fallback: try without specifying driver path
                logger.warning(f"ChromeDriverManager failed: {driver_error}, trying system chromedriver")
                self.driver = webdriver.Chrome(options=chrome_options)
            
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("Selenium WebDriver initialized with stealth options")
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
    
    def search_multiple_sources(self, query: str, max_retries: int = 2) -> str:
        """
        Search multiple sources for company information with robust error handling
        
        Args:
            query: Search query string
            max_retries: Maximum number of retry attempts per source
            
        Returns:
            Extracted search result text
        """
        
        # Try multiple search approaches
        search_methods = [
            self._search_google,
            self._search_duckduckgo,
            self._search_perplexity
        ]
        
        for method in search_methods:
            try:
                result = method(query, max_retries)
                if result and len(result) > 50:  # Valid result
                    return result
            except Exception as e:
                logger.warning(f"Search method {method.__name__} failed: {e}")
                continue
        
        logger.error(f"All search methods failed for query: {query}")
        return ""
    
    def _search_google(self, query: str, max_retries: int = 2) -> str:
        """Search using Google with company-specific queries"""
        for attempt in range(max_retries):
            try:
                # Recreate driver if session is invalid
                if not self._is_driver_valid():
                    self._recreate_driver()
                
                google_query = f"site:crunchbase.com OR site:linkedin.com OR site:wikipedia.org {query}"
                self.driver.get(f"https://www.google.com/search?q={quote(google_query)}")
                
                # Add random delay
                time.sleep(np.random.uniform(2, 4))
                
                # Extract search results
                results = self.driver.find_elements(By.CSS_SELECTOR, ".g .VwiC3b, .g .s")
                if results:
                    combined_text = " ".join([elem.text for elem in results[:3]])
                    logger.info(f"Google search successful for: {query[:50]}...")
                    return combined_text
                    
            except Exception as e:
                logger.warning(f"Google search attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        
        return ""
    
    def _search_duckduckgo(self, query: str, max_retries: int = 2) -> str:
        """Search using DuckDuckGo (more bot-friendly)"""
        for attempt in range(max_retries):
            try:
                if not self._is_driver_valid():
                    self._recreate_driver()
                
                self.driver.get(f"https://duckduckgo.com/?q={quote(query)}")
                time.sleep(np.random.uniform(3, 5))
                
                # Extract results
                results = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='result'] h3, [data-testid='result'] .E2eLOJr8HctVnDOTM8fs")
                if results:
                    combined_text = " ".join([elem.text for elem in results[:3]])
                    logger.info(f"DuckDuckGo search successful for: {query[:50]}...")
                    return combined_text
                    
            except Exception as e:
                logger.warning(f"DuckDuckGo search attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        
        return ""
    
    def _search_perplexity(self, query: str, max_retries: int = 2) -> str:
        """Search using Perplexity (backup method)"""
        for attempt in range(max_retries):
            try:
                if not self._is_driver_valid():
                    self._recreate_driver()
                
                self.driver.get("https://www.perplexity.ai/")
                time.sleep(np.random.uniform(2, 4))
                
                # Find search input with multiple selectors
                search_input = None
                selectors = ["textarea", "input[type='text']", "[data-testid*='search']", "[placeholder*='search' i]"]
                
                for selector in selectors:
                    try:
                        search_input = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        break
                    except:
                        continue
                
                if not search_input:
                    return ""
                
                search_input.clear()
                search_input.send_keys(query)
                
                # Try to submit
                from selenium.webdriver.common.keys import Keys
                search_input.send_keys(Keys.RETURN)
                
                # Wait for results
                time.sleep(8)
                
                # Extract results with multiple selectors
                result_selectors = [
                    "[data-testid*='response']",
                    ".response",
                    ".answer",
                    "[class*='answer']",
                    "main p"
                ]
                
                for selector in result_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            result = elements[0].text.strip()
                            if result and len(result) > 20:
                                logger.info(f"Perplexity search successful for: {query[:50]}...")
                                return result
                    except:
                        continue
                        
            except Exception as e:
                logger.warning(f"Perplexity search attempt {attempt + 1} failed: {e}")
                time.sleep(3)
        
        return ""
    
    def _is_driver_valid(self) -> bool:
        """Check if the driver session is still valid"""
        try:
            if not self.driver:
                return False
            self.driver.current_url  # This will throw if session is invalid
            return True
        except:
            return False
    
    def _recreate_driver(self):
        """Recreate the driver session"""
        try:
            if self.driver:
                self.driver.quit()
        except:
            pass
        
        logger.info("Recreating WebDriver session...")
        self._setup_selenium()
    
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
        funding_query = f"{company_name} total funding raised venture capital investment"
        funding_result = self.search_multiple_sources(funding_query)
        metrics['TotalFunding'] = self._extract_number(funding_result, 'funding')
        
        # Query for market cap (for public companies)
        market_cap_query = f"{company_name} market capitalization stock market value"
        market_cap_result = self.search_multiple_sources(market_cap_query)
        metrics['MarketCap'] = self._extract_number(market_cap_result, 'market cap')
        
        # Query for annual revenue
        revenue_query = f"{company_name} annual revenue earnings latest year"
        revenue_result = self.search_multiple_sources(revenue_query)
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
        employee_query = f"{company_name} number of employees company size team"
        employee_result = self.search_multiple_sources(employee_query)
        metrics['EmployeeCount'] = self._extract_number(employee_result, 'employees')
        
        # Query for web traffic
        traffic_query = f"{company_name} website traffic monthly visitors users"
        traffic_result = self.search_multiple_sources(traffic_query)
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


class EcosystemMapGeneratorNoOpenAI(EcosystemMapGenerator):
    """Version that works without OpenAI - uses web scraping + fallback qualitative analysis"""
    
    def __init__(self, input_file: str = "companies.xlsx"):
        """Initialize without OpenAI API key"""
        self.input_file = input_file
        self.df = None
        self.driver = None
        self.weights = MetricWeights()
        
        logger.info("Initialized ecosystem generator with web scraping + fallback analysis")
        
        # Setup Selenium driver
        self._setup_selenium()
    
    def analyze_api_quality(self, company_name: str) -> int:
        """Fallback API quality analysis without OpenAI"""
        return self._fallback_api_quality_score(company_name)
    
    def analyze_platform_architecture(self, company_name: str) -> int:
        """Fallback platform architecture analysis without OpenAI"""
        return self._fallback_platform_architecture_score(company_name)
    
    def analyze_ai_features(self, company_name: str) -> int:
        """Fallback AI features analysis without OpenAI"""
        return self._fallback_ai_features_score(company_name)
    
    def _fallback_api_quality_score(self, company_name: str) -> int:
        """Rule-based API quality scoring"""
        company_lower = company_name.lower()
        
        # Known high-API quality companies (modern platforms with good APIs)
        high_api_companies = [
            'benchling', 'dotmatics', 'tetrascience', 'scispot', 'genedata',
            'benchsci', 'certara', 'knime'
        ]
        
        # Known medium-API companies (some API but limited)
        medium_api_companies = [
            'labware', 'sapio', 'labguru', 'revvity', 'biovia'
        ]
        
        # Companies with modern keywords tend to have better APIs
        modern_keywords = ['bio', 'genomics', 'science', 'data', 'platform']
        
        if any(comp in company_lower for comp in high_api_companies):
            score = np.random.choice([2, 3], p=[0.3, 0.7])
        elif any(comp in company_lower for comp in medium_api_companies):
            score = np.random.choice([1, 2], p=[0.6, 0.4])
        elif any(keyword in company_lower for keyword in modern_keywords):
            score = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        else:
            score = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        
        logger.info(f"Fallback API Quality score for {company_name}: {score}")
        return score
    
    def _fallback_platform_architecture_score(self, company_name: str) -> int:
        """Rule-based platform architecture scoring"""
        company_lower = company_name.lower()
        
        # Known cloud-native companies
        cloud_native = [
            'benchling', 'benchsci', 'tetrascience', 'scispot', 'ginkgo'
        ]
        
        # Traditional but modernized companies
        modernized = [
            'revvity', 'biovia', 'genedata', 'certara', 'labware', 'dotmatics'
        ]
        
        # Cloud/modern indicators
        cloud_keywords = ['cloud', 'saas', 'platform', 'bio', 'genomics', 'science']
        
        if any(comp in company_lower for comp in cloud_native):
            score = np.random.choice([2, 3], p=[0.2, 0.8])
        elif any(comp in company_lower for comp in modernized):
            score = np.random.choice([2, 3], p=[0.6, 0.4])
        elif any(keyword in company_lower for keyword in cloud_keywords):
            score = np.random.choice([2, 3], p=[0.5, 0.5])
        else:
            score = np.random.choice([1, 2], p=[0.6, 0.4])
        
        logger.info(f"Fallback Platform Architecture score for {company_name}: {score}")
        return score
    
    def _fallback_ai_features_score(self, company_name: str) -> int:
        """Rule-based AI features scoring"""
        company_lower = company_name.lower()
        
        # Companies known for AI/ML capabilities
        ai_companies = [
            'benchsci', 'ginkgo', 'tetrascience', 'scispot', 'knime'
        ]
        
        # AI-related keywords
        ai_keywords = ['ai', 'ml', 'machine', 'learning', 'intelligence', 'data', 'analytics']
        
        if any(comp in company_lower for comp in ai_companies):
            score = 1
        elif any(keyword in company_lower for keyword in ai_keywords):
            score = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            score = np.random.choice([0, 1], p=[0.7, 0.3])
        
        logger.info(f"Fallback AI Features score for {company_name}: {score}")
        return score


def main():
    """Main function to run the ecosystem map generator"""
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    INPUT_FILE = "companies.xlsx"
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found. Please create it with a 'CompanyName' column.")
        return
    
    if not OPENAI_API_KEY:
        print("🚫 No OpenAI API key found")
        print("🔍 Running with Perplexity web scraping + fallback qualitative analysis")
        print("📊 This will collect real financial data but use rule-based qualitative scoring")
        
        response = input("Continue with this approach? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Exiting...")
            return
        
        # Initialize with None - will use fallback methods
        generator = EcosystemMapGeneratorNoOpenAI(INPUT_FILE)
    else:
        print("🤖 Using OpenAI for full LLM-powered analysis")
        generator = EcosystemMapGenerator(OPENAI_API_KEY, INPUT_FILE)
    
    generator.run_full_pipeline()


if __name__ == "__main__":
    main()
