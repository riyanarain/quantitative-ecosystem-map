# Quantitative Ecosystem Map Generator

A Python script that generates a "Quantitative Ecosystem Map" by analyzing companies through automated web scraping and AI reasoning, calculating scores based on a defined methodology, and outputting a professional 2D scatter plot visualization.

## Overview

The script creates a comprehensive analysis of companies across two dimensions:
- **X-axis (Dominance)**: Financial strength and market penetration metrics
- **Y-axis (AI Ready)**: API quality, platform architecture, and native AI features

**Pipeline in One Line:**
```
📊 companies.xlsx → 🌐 Web Scraping → 🤖 AI Analysis → ⚖️ Scoring → 📈 ecosystem_map.png
```

## Pipeline Architecture

```
📊 INPUT DATA
    │
    ├── companies.xlsx (17 companies)
    │
    ▼
🌐 MULTI-SOURCE WEB SCRAPING
    │
    ├── Google Search ──────────┐
    ├── DuckDuckGo ─────────────┤
    ├── Perplexity AI ──────────┤
    │                           │
    ▼                           ▼
💰 FINANCIAL METRICS        🏢 MARKET METRICS
    │                           │
    ├── Total Funding           ├── Employee Count
    ├── Market Cap              ├── Web Traffic
    ├── Annual Revenue          │
    │                           │
    ▼                           ▼
🤖 AI-POWERED QUALITATIVE ANALYSIS
    │
    ├── OpenAI GPT-4 ───────────┐
    │   ├── API Quality Score    │
    │   ├── Platform Architecture│
    │   └── AI Features Score    │
    │                            │
    ├────────────────────────────
    │
    ▼
⚖️ NORMALIZATION & SCORING
    │
    ├── Min-Max Scaling (0-100)
    ├── Weighted Averages
    │   ├── X-axis: Financial(0.7) + Market(0.3)
    │   └── Y-axis: API(0.5) + Platform(0.3) + AI(0.2)
    │
    ▼
📈 VISUALIZATION & OUTPUT
    │
    ├── ecosystem_map.png
    ├── ecosystem_data_final.xlsx
    └── Progress checkpoints
```

## Features

- **Automated Web Scraping**: Multi-source data collection from Google, DuckDuckGo, and Perplexity
- **AI-Powered Analysis**: OpenAI GPT-4 integration for qualitative scoring with intelligent fallbacks
- **Robust Data Collection**: Stealth browsing, session recovery, and error handling
- **Professional Visualization**: High-quality scatter plots with company positioning
- **Demo Results Available**: Pre-generated results from full analysis pipeline
- **Configurable Weights**: Easily adjust metric importance

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
3. (Optional) Add your OpenAI API key to `.env`:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```
4. Run the full analysis:
   ```bash
   python3 ecosystem_map.py
   ```

### Manual Installation

If you prefer to install manually:

```bash
pip install -r requirements.txt
```

## Usage

### Full Analysis Pipeline

Run the complete analysis with web scraping and AI reasoning:

```bash
python3 ecosystem_map.py
```

The script executes a comprehensive 7-stage pipeline:

**Stage 1: Data Loading**
- Load company list from `companies.xlsx`
- Initialize web scraping session with stealth options
- Set up progress tracking and error recovery

**Stage 2: Financial Data Collection**
- Search multiple sources for funding information
- Extract market capitalization for public companies
- Gather annual revenue data
- Calculate financial strength composite score

**Stage 3: Market Penetration Analysis**
- Collect employee count and company size metrics
- Extract web traffic and digital engagement data
- Calculate market penetration composite score

**Stage 4: AI-Powered Qualitative Assessment**
- Analyze API quality and documentation (0-3 scale)
- Evaluate platform architecture and technology stack (1-3 scale)
- Assess native AI/ML capabilities (0-1 scale)
- Apply OpenAI GPT-4 analysis or intelligent fallback scoring

**Stage 5: Data Normalization**
- Apply min-max scaling to all metrics (0-100 range)
- Handle missing data and outliers
- Ensure consistent scoring across all companies

**Stage 6: Weighted Scoring**
- Calculate X-axis (Dominance): Financial Strength (70%) + Market Penetration (30%)
- Calculate Y-axis (AI Ready): API Score (50%) + Platform Architecture (30%) + AI Features (20%)
- Generate final positioning coordinates

**Stage 7: Visualization & Export**
- Create professional 2D scatter plot
- Position companies with intelligent label placement
- Export visualization and complete dataset
- Save progress checkpoints for recovery

### Company Data

The analysis uses `companies.xlsx` with a single column called `CompanyName`:

| CompanyName |
|-------------|
| Benchling   |
| BenchSci    |
| Biovia      |
| ...         |

A sample file with 17 biotech companies is included.

### Demo Results (Pre-Generated)

To view results from a previous full analysis run:

```bash
python3 demo_with_mock_data.py
```

This displays results generated from the full web scraping and reasoning pipeline, showing:
- `demo_ecosystem_map.png` - Professional visualization based on actual data collection
- `demo_ecosystem_data.xlsx` - Dataset derived from web scraping and AI analysis

### Output Files

The full analysis generates:
- `ecosystem_map.png` - The final visualization
- `ecosystem_data_final.xlsx` - Complete dataset with all metrics
- `ecosystem_data_progress_X.xlsx` - Progress checkpoints (auto-saved)

## Methodology

### Quantitative Metrics (X-axis: Dominance)

**Financial Strength (70% weight)**:
- Total funding raised (private companies)
- Market capitalization (public companies)  
- Annual revenue

**Market Penetration (30% weight)**:
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

```
📊 RAW DATA COLLECTION
    │
    ├── 💰 Financial Data          🏢 Market Data          🤖 Qualitative Data
    │   ├── Funding: $XM           ├── Employees: X        ├── API Quality: 0-3
    │   ├── Market Cap: $XM        ├── Traffic: X visits   ├── Platform Arch: 1-3
    │   └── Revenue: $XM           └── Market Score        └── AI Features: 0-1
    │                                                      
    ▼                              ▼                       ▼
⚖️ NORMALIZATION (0-100 SCALE)
    │
    ├── Financial Strength         Market Penetration      AI Readiness Components
    │   Min-Max Scaled            Min-Max Scaled          Min-Max Scaled
    │                                                      
    ▼                              ▼                       ▼
🎯 WEIGHTED SCORING
    │
    ├── X-AXIS (DOMINANCE)                    Y-AXIS (AI READY)
    │   ├── Financial Strength × 0.7          ├── API Quality × 0.5
    │   └── Market Penetration × 0.3          ├── Platform Arch × 0.3
    │                                         └── AI Features × 0.2
    ▼                                         ▼
📈 FINAL POSITIONING (X, Y coordinates for 2D plot)
```

**Processing Steps:**
1. **Data Collection**: Multi-source web scraping and LLM analysis
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
        'FinancialStrength': 0.7,
        'MarketPenetration': 0.3,
    }
    y_axis = {
        'APIScore': 0.5,
        'PlatformArchitecture': 0.3,
        'NativeAIFeatures': 0.2,
    }
```

### Technical Architecture

**Multi-Source Data Collection**:
- **Google Search**: Company information from authoritative sources
- **DuckDuckGo**: Bot-friendly alternative with reliable results  
- **Perplexity AI**: Advanced search with contextual understanding
- **Fallback Chain**: Automatic source switching on failures

**Stealth Web Scraping**:
- Randomized user agents and browser fingerprints
- Anti-detection measures (remove automation flags)
- Session recovery and driver recreation
- Smart rate limiting with jittered delays

**AI-Powered Analysis**:
- OpenAI GPT-4 integration for qualitative assessment
- Intelligent fallback scoring when API unavailable
- Context-aware prompts tailored to each metric
- Rule-based analysis with industry knowledge

**Enterprise-Grade Reliability**:
- Progress checkpoints every 5 companies
- Automatic error recovery and retry logic
- Cross-platform compatibility (Windows, macOS, Linux)
- Graceful degradation on component failures

## Troubleshooting

### Common Issues

**ChromeDriver issues**:
- The script automatically downloads and manages ChromeDriver
- Ensure Chrome browser is installed and up-to-date
- On macOS ARM, the script includes special handling for driver paths

**OpenAI API errors**:
- Verify your API key is correct in `.env`
- The script works without OpenAI (uses intelligent fallback scoring)
- Check your OpenAI account has sufficient credits if using GPT-4

**Web scraping challenges**:
- The script tries multiple sources (Google, DuckDuckGo, Perplexity)
- Built-in stealth options and session recovery
- Some data extraction failures are normal and handled gracefully

**Package installation errors**:
- Ensure you have Python 3.8+ installed
- Try running `python3 -m pip install -r requirements.txt`

**Memory issues with large datasets**:
- Use the progress save feature to resume interrupted runs
- Process companies in smaller batches if needed

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

### Adding New Companies

1. Edit `companies.xlsx` to add/remove companies
2. Run the full analysis to collect new data

### Modifying Analysis

1. **Web Scraping Sources**: Edit search methods in `ecosystem_map.py`
2. **LLM Prompts**: Modify prompt templates in analysis methods
3. **Metric Weights**: Update the `MetricWeights` configuration
4. **Scoring Logic**: Adjust normalization and calculation methods

### Alternative Data Sources

The script is designed to be extensible:
- Add new web scraping targets
- Integrate additional APIs
- Implement custom data collection methods

### Visualization Customization

The script uses matplotlib/seaborn. You can modify the `create_visualization()` method to change:
- Plot style and colors
- Company positioning algorithms
- Label placement and formatting

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all prerequisites are properly installed (Python, Chrome, optional OpenAI key)
3. Ensure your environment configuration is correct

## Sample Companies

The included sample dataset focuses on 17 biopharma data platform companies:
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
- Revvity
- Sapio
- SciNote
- Scispot
- SimulationsPlus
- tetrascience

You can replace this with any set of companies relevant to your analysis.
