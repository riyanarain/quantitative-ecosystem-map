# Quantitative Ecosystem Map Generator

A Python script that generates a "Quantitative Ecosystem Map" by analyzing companies through web scraping and LLM reasoning, calculating scores based on a defined methodology, and outputting a 2D scatter plot.

## Overview

The script creates a comprehensive analysis of companies across two dimensions:
- **X-axis (Dominance)**: Financial strength and market penetration metrics
- **Y-axis (AI Ready)**: API quality, platform architecture, and native AI features

## Features

- **Automated Data Collection**: Web scraping using Selenium and Perplexity AI
- **LLM Analysis**: OpenAI GPT-4 integration for qualitative scoring
- **Data Processing**: Normalization, weighting, and score calculation
- **Visualization**: Professional scatter plots with company labels

## Installation

### Prerequisites

1. **Python 3.8+** installed
2. **Chrome browser** installed (for web scraping)
3. **OpenAI API key** (for LLM analysis)

### Quick Setup

1. Clone or download this repository
2. Run the setup script:
   ```bash
   python3 setup.py
   ```
3. Edit the `.env` file and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Manual Installation

If you prefer to install manually:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data

Create an Excel file named `companies.xlsx` with a single column called `CompanyName`:

| CompanyName |
|-------------|
| Benchling   |
| BenchSci    |
| Biovia      |
| ...         |

A sample file with 20 biotech companies is already included.

### 2. Set Environment Variables

Create a `.env` file (or edit the existing one):

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Analysis

```bash
python3 ecosystem_map.py
```

The script will:
1. Load companies from `companies.xlsx`
2. Scrape web data for each company using Perplexity AI
3. Analyze qualitative metrics using OpenAI GPT-4
4. Calculate normalized scores and final positioning
5. Generate and save the ecosystem map visualization

### 4. Output Files

- `ecosystem_map.png` - The final visualization
- `ecosystem_data_final.xlsx` - Complete dataset with all metrics
- `ecosystem_data_progress_X.xlsx` - Progress checkpoints (auto-saved)

## Methodology

### Quantitative Metrics (X-axis: Dominance)

**Financial Strength (40% weight)**:
- Total funding raised (private companies)
- Market capitalization (public companies)  
- Annual revenue

**Market Penetration (60% weight)**:
- Number of employees
- Web traffic and engagement

### Qualitative Metrics (Y-axis: AI Ready)

**API Quality Score (50% weight)**:
- 0: No public API found
- 1: Limited or poorly documented API
- 2: Comprehensive, well-documented REST API
- 3: Comprehensive API + dedicated Python SDK

**Platform Architecture Score (30% weight)**:
- 1: Legacy on-premise software
- 2: Cloud-enabled or hosted legacy software
- 3: Modern, multi-tenant, cloud-native SaaS

**Native AI/ML Features (20% weight)**:
- 0: No native AI capabilities
- 1: Native AI/ML features available

### Scoring Process

1. **Data Collection**: Web scraping and LLM analysis
2. **Normalization**: Min-max scaling to 0-100 range
3. **Weighting**: Applying configured weights to each metric
4. **Final Scores**: Weighted averages for X and Y axes

## Configuration

### Adjusting Weights

Edit the `MetricWeights` class in `ecosystem_map.py`:

```python
@dataclass
class MetricWeights:
    x_axis = {
        'FinancialStrength': 0.4,
        'MarketPenetration': 0.6,
    }
    y_axis = {
        'APIScore': 0.5,
        'PlatformArchitecture': 0.3,
        'NativeAIFeatures': 0.2,
    }
```

### Rate Limiting

The script includes built-in rate limiting:
- 1-second delays between OpenAI API calls
- 2-5 second delays between web scraping requests
- Retry logic for failed requests

### Progress Saving

Progress is automatically saved every 5 companies to prevent data loss during long runs.

## Troubleshooting

### Common Issues

**ChromeDriver not found**:
- The script will automatically download ChromeDriver using webdriver-manager
- Ensure Chrome browser is installed

**OpenAI API errors**:
- Verify your API key is correct in `.env`
- Check your OpenAI account has sufficient credits
- Ensure you have access to GPT-4 (or modify the model in the script)

**Perplexity scraping fails**:
- The script includes retry logic and fallback mechanisms
- Some failures are expected and handled gracefully

**Memory issues with large datasets**:
- Process companies in smaller batches
- Use the progress save feature to resume interrupted runs

### Debugging

Enable debug logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Example Output

The final visualization shows companies positioned on a 2D map:
- **Bottom-left**: Low dominance, low AI readiness
- **Top-right**: High dominance, high AI readiness
- **Top-left**: Low dominance, high AI readiness (emerging AI-native companies)
- **Bottom-right**: High dominance, low AI readiness (established traditional companies)

## Customization

### Adding New Metrics

1. Add data collection methods in the appropriate class methods
2. Update the `MetricWeights` configuration
3. Modify the normalization and scoring logic as needed

### Changing LLM Prompts

Edit the prompt templates in methods like `analyze_api_quality()` to adjust the analysis criteria.

### Alternative Visualization

The script uses matplotlib/seaborn. You can modify the `create_visualization()` method to change the plot style, colors, or layout.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all prerequisites are properly installed
3. Ensure your `.env` file is correctly configured

## Sample Companies

The included sample dataset focuses on biopharma data platform companies:
- Benchling
- BenchSci
- Biovia
- CDD Vault
- Certara
- Dotmatics
- Genedata
- Ginkgo Bioworks
- KNIME
- Labguru
- LabWare
- LiveDesign (Schrödinger)
- ReSync
- Revvity
- Sapio
- SciNote
- Scispot
- SimulationsPlus
- tetrascience
- Thermo Scientific


You can replace this with any set of companies relevant to your analysis.
