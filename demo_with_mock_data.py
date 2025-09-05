#!/usr/bin/env python3
"""
Demo script for the Ecosystem Map Generator using mock data.
This allows testing the visualization and data processing without running web scraping or API calls.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass

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

def create_mock_data():
    """Create realistic mock data for demonstration"""
    
    companies = [
        "Benchling", "BenchSci", "Biovia", "CDD Vault", "Certara",
        "Dotmatics", "Genedata", "Ginkgo Bioworks", "KNIME", "Labguru",
        "LabWare", "LiveDesign (Schrödinger)", "ReSync", "Revvity", "Sapio",
        "SciNote", "Scispot", "SimulationsPlus", "tetrascience", "Thermo Scientific"
    ]
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    data = []
    for company in companies:
        # Generate realistic mock metrics
        
        # Financial metrics (varied by company type/maturity)
        if company in ["Thermo Scientific", "Revvity", "Biovia"]:
            # Large public/established companies
            total_funding = np.random.uniform(500, 2000)  # millions
            market_cap = np.random.uniform(5000, 50000)   # millions
            revenue = np.random.uniform(1000, 5000)       # millions
            financial_strength = max(total_funding, market_cap, revenue)
        elif company in ["Benchling", "Dotmatics", "Genedata", "LabWare", "Certara"]:
            # Mid-size established platforms
            total_funding = np.random.uniform(200, 800)
            market_cap = np.random.uniform(1000, 5000)
            revenue = np.random.uniform(100, 1000)
            financial_strength = max(total_funding, market_cap, revenue)
        else:
            # Smaller/emerging platforms
            total_funding = np.random.uniform(50, 500)
            market_cap = 0  # Private
            revenue = np.random.uniform(10, 200)
            financial_strength = max(total_funding, revenue)
        
        # Market penetration metrics
        if company in ["Thermo Scientific", "Revvity", "Biovia"]:
            employee_count = np.random.uniform(5000, 20000)
            web_traffic = np.random.uniform(500000, 2000000)  # monthly visitors
        elif company in ["Benchling", "Dotmatics", "Genedata", "LabWare", "KNIME"]:
            employee_count = np.random.uniform(1000, 5000)
            web_traffic = np.random.uniform(100000, 500000)
        else:
            employee_count = np.random.uniform(100, 1000)
            web_traffic = np.random.uniform(10000, 100000)
        
        # Calculate market penetration score
        employee_score = min(employee_count / 10, 100)
        traffic_score = min(web_traffic / 100000, 100)
        market_penetration = max(employee_score, traffic_score)
        
        # Qualitative AI-Ready metrics
        if company in ["BenchSci", "tetrascience", "Scispot", "LiveDesign (Schrödinger)"]:
            # AI-native and modern cloud platforms
            api_score = np.random.choice([2, 3], p=[0.3, 0.7])
            platform_architecture = np.random.choice([2, 3], p=[0.2, 0.8])
            native_ai_features = 1
        elif company in ["Benchling", "Dotmatics", "KNIME", "Genedata", "Certara"]:
            # Modern scientific platforms
            api_score = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            platform_architecture = np.random.choice([2, 3], p=[0.4, 0.6])
            native_ai_features = np.random.choice([0, 1], p=[0.6, 0.4])
        else:
            # Traditional/legacy platforms
            api_score = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            platform_architecture = np.random.choice([1, 2], p=[0.6, 0.4])
            native_ai_features = np.random.choice([0, 1], p=[0.8, 0.2])
        
        data.append({
            'CompanyName': company,
            'TotalFunding': total_funding,
            'MarketCap': market_cap,
            'Revenue': revenue,
            'FinancialStrength': financial_strength,
            'EmployeeCount': employee_count,
            'WebTraffic': web_traffic,
            'MarketPenetration': market_penetration,
            'APIScore': api_score,
            'PlatformArchitecture': platform_architecture,
            'NativeAIFeatures': native_ai_features
        })
    
    return pd.DataFrame(data)

def normalize_data(df):
    """Normalize all metric columns to 0-100 scale"""
    metric_columns = ['FinancialStrength', 'MarketPenetration', 'APIScore', 'PlatformArchitecture', 'NativeAIFeatures']
    
    for col in metric_columns:
        if col in df.columns:
            # Min-max normalization to 0-100 scale
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max > col_min:
                df[f'{col}_Normalized'] = 100 * (df[col] - col_min) / (col_max - col_min)
            else:
                df[f'{col}_Normalized'] = 50  # If all values are the same
                
            logger.info(f"Normalized {col}: min={col_min:.2f}, max={col_max:.2f}")
    
    return df

def calculate_final_scores(df, weights):
    """Calculate weighted final scores for X and Y axes"""
    
    # Calculate X-axis score (Dominance)
    x_components = []
    for metric, weight in weights.x_axis.items():
        col_name = f'{metric}_Normalized'
        if col_name in df.columns:
            x_components.append(df[col_name] * weight)
    
    df['Dominance_Score'] = sum(x_components)
    
    # Calculate Y-axis score (AI Ready)
    y_components = []
    for metric, weight in weights.y_axis.items():
        col_name = f'{metric}_Normalized'
        if col_name in df.columns:
            y_components.append(df[col_name] * weight)
    
    df['AI_Ready_Score'] = sum(y_components)
    
    logger.info("Final scores calculated successfully")
    return df

def create_visualization(df, output_file="demo_ecosystem_map.png"):
    """Create and save the ecosystem map visualization"""
    
    # Set up the plot style
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create color-coded scatter plot based on AI readiness
    colors = []
    for score in df['AI_Ready_Score']:
        if score >= 70:
            colors.append('#2E8B57')  # Dark green for high AI readiness
        elif score >= 40:
            colors.append('#4682B4')  # Steel blue for medium AI readiness
        else:
            colors.append('#CD853F')  # Peru for low AI readiness
    
    # Create scatter plot
    scatter = ax.scatter(
        df['Dominance_Score'],
        df['AI_Ready_Score'],
        s=120,
        alpha=0.7,
        c=colors,
        edgecolors='black',
        linewidth=0.8
    )
    
    # Add company labels with improved positioning
    for idx, row in df.iterrows():
        ax.annotate(
            row['CompanyName'],
            (row['Dominance_Score'], row['AI_Ready_Score']),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    # Customize the plot
    ax.set_xlabel('Dominance in the Ecosystem', fontsize=16, fontweight='bold')
    ax.set_ylabel('AI Ready', fontsize=16, fontweight='bold')
    ax.set_title('Quantitative Ecosystem Map - Laboratory Informatics & Scientific Software\n(Demo with Mock Data)', 
                 fontsize=18, fontweight='bold', pad=25)
    
    # Add quadrant labels
    ax.text(0.05, 0.95, 'Emerging\nAI Leaders', transform=ax.transAxes, fontsize=12, 
            ha='left', va='top', style='italic', alpha=0.7)
    ax.text(0.95, 0.95, 'Dominant\nAI Leaders', transform=ax.transAxes, fontsize=12, 
            ha='right', va='top', style='italic', alpha=0.7)
    ax.text(0.05, 0.05, 'Traditional\nPlayers', transform=ax.transAxes, fontsize=12, 
            ha='left', va='bottom', style='italic', alpha=0.7)
    ax.text(0.95, 0.05, 'Established\nPlayers', transform=ax.transAxes, fontsize=12, 
            ha='right', va='bottom', style='italic', alpha=0.7)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with padding
    x_margin = (df['Dominance_Score'].max() - df['Dominance_Score'].min()) * 0.1
    y_margin = (df['AI_Ready_Score'].max() - df['AI_Ready_Score'].min()) * 0.1
    
    ax.set_xlim(
        df['Dominance_Score'].min() - x_margin,
        df['Dominance_Score'].max() + x_margin
    )
    ax.set_ylim(
        df['AI_Ready_Score'].min() - y_margin,
        df['AI_Ready_Score'].max() + y_margin
    )
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', label='High AI Readiness (70+)'),
        Patch(facecolor='#4682B4', label='Medium AI Readiness (40-70)'),
        Patch(facecolor='#CD853F', label='Lower AI Readiness (<40)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.85))
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Demo ecosystem map saved as {output_file}")

def print_summary_statistics(df):
    """Print summary statistics of the analysis"""
    print("\n" + "="*60)
    print("ECOSYSTEM ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nTotal Companies Analyzed: {len(df)}")
    
    print(f"\nDominance Scores:")
    print(f"  Average: {df['Dominance_Score'].mean():.1f}")
    print(f"  Range: {df['Dominance_Score'].min():.1f} - {df['Dominance_Score'].max():.1f}")
    
    print(f"\nAI Readiness Scores:")
    print(f"  Average: {df['AI_Ready_Score'].mean():.1f}")
    print(f"  Range: {df['AI_Ready_Score'].min():.1f} - {df['AI_Ready_Score'].max():.1f}")
    
    # Top performers
    print(f"\nTop 5 Most Dominant Companies:")
    top_dominant = df.nlargest(5, 'Dominance_Score')[['CompanyName', 'Dominance_Score']]
    for idx, row in top_dominant.iterrows():
        print(f"  {row['CompanyName']}: {row['Dominance_Score']:.1f}")
    
    print(f"\nTop 5 Most AI-Ready Companies:")
    top_ai = df.nlargest(5, 'AI_Ready_Score')[['CompanyName', 'AI_Ready_Score']]
    for idx, row in top_ai.iterrows():
        print(f"  {row['CompanyName']}: {row['AI_Ready_Score']:.1f}")
    
    # Quadrant analysis
    high_dom = df['Dominance_Score'] > df['Dominance_Score'].median()
    high_ai = df['AI_Ready_Score'] > df['AI_Ready_Score'].median()
    
    quadrants = {
        'Leaders (High Dom, High AI)': df[high_dom & high_ai],
        'Emerging (Low Dom, High AI)': df[~high_dom & high_ai],
        'Established (High Dom, Low AI)': df[high_dom & ~high_ai],
        'Traditional (Low Dom, Low AI)': df[~high_dom & ~high_ai]
    }
    
    print(f"\nQuadrant Analysis:")
    for quad_name, quad_companies in quadrants.items():
        print(f"  {quad_name}: {len(quad_companies)} companies")
        if len(quad_companies) > 0:
            companies = ', '.join(quad_companies['CompanyName'].tolist()[:3])
            if len(quad_companies) > 3:
                companies += f" (and {len(quad_companies)-3} more)"
            print(f"    Examples: {companies}")

def main():
    """Run the demo with mock data"""
    print("Ecosystem Map Generator - Demo with Mock Data")
    print("=" * 50)
    
    # Step 1: Create mock data
    logger.info("Creating mock data...")
    df = create_mock_data()
    logger.info(f"Created data for {len(df)} companies")
    
    # Step 2: Normalize data
    logger.info("Normalizing data...")
    df = normalize_data(df)
    
    # Step 3: Calculate final scores
    logger.info("Calculating final scores...")
    weights = MetricWeights()
    df = calculate_final_scores(df, weights)
    
    # Step 4: Create visualization
    logger.info("Creating visualization...")
    create_visualization(df)
    
    # Step 5: Save results
    output_file = "demo_ecosystem_data.xlsx"
    df.to_excel(output_file, index=False)
    logger.info(f"Demo data saved to {output_file}")
    
    # Step 6: Print summary
    print_summary_statistics(df)
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print(f"Files created:")
    print(f"  - demo_ecosystem_map.png (visualization)")
    print(f"  - demo_ecosystem_data.xlsx (full dataset)")
    print(f"\nThis demo used mock data. To run with real data:")
    print(f"1. Add your OpenAI API key to .env file")
    print(f"2. Run: python3 ecosystem_map.py")

if __name__ == "__main__":
    main()
