# Quantitative Ecosystem Map Generator

A Python script that generates a "Quantitative Ecosystem Map" by analyzing companies through automated web scraping and AI reasoning, calculating scores based on a defined methodology, and outputting a professional 2D scatter plot visualization.

## Overview

The script creates a comprehensive analysis of companies across two key dimensions to map a competitive landscape.

**Pipeline in One Line:**

```
📊 companies.xlsx → 🌐 Web Scraping → 🤖 AI Analysis → ⚖️ Scoring → 📈 ecosystem_map.png
```

## Features

  - **Automated Data Collection**: Gathers data from multiple web sources (Google, DuckDuckGo, Perplexity).
  - **AI-Powered Analysis**: Integrates OpenAI GPT-4 for nuanced qualitative scoring.
  - **Robust & Reliable**: Features stealth browsing, error handling, and session recovery checkpoints.
  - **Professional Visualization**: Creates high-quality, labeled scatter plots.
  - **Configurable Weights**: Easily adjust the importance of each metric to tune the analysis.

-----

## Installation & Usage

### Prerequisites

1.  **Python 3.8+**
2.  **Chrome browser**
3.  **OpenAI API key** (optional, the script has fallbacks)

### Quick Setup & Run

1.  Clone or download this repository.
2.  Run the setup script to install dependencies:
    ```bash
    python3 setup.py
    ```
3.  (Optional) Add your OpenAI API key to a new `.env` file:
    ```bash
    OPENAI_API_KEY=your_openai_api_key_here
    ```
4.  Run the full analysis pipeline:
    ```bash
    python3 ecosystem_map.py
    ```

### Demo with Pre-Generated Results

To view a map generated from a previous full analysis without running the scrapers, use the mock data script:

```bash
python3 demo_with_mock_data.py
```

-----

## Methodology & Pipeline

This section details the logic for scoring and plotting each company. The script executes a comprehensive 7-stage pipeline to generate the final map.

### Stage 1: Data Loading

  - Loads the company list from `companies.xlsx`.
  - Initializes a web scraping session and sets up progress tracking.

### Stage 2: Financial Data Collection (X-Axis)

  - Searches multiple sources for funding, market capitalization, and annual revenue.

### Stage 3: Market Penetration Analysis (X-Axis)

  - Collects employee count, company size, and web traffic data.

### Stage 4: AI-Powered Qualitative Assessment (Y-Axis)

  - Uses OpenAI GPT-4 (or intelligent fallbacks) to analyze and score API quality, platform architecture, and native AI features against a defined rubric.

### Stage 5: Data Normalization

  - Applies min-max scaling to all metrics to convert them to a uniform 0-100 range.

### Stage 6: Weighted Scoring

  - **X-Axis (Dominance)** is calculated as:
      - Financial Strength (70%) + Market Penetration (30%)
  - **Y-Axis (AI Ready)** is calculated as:
      - API Score (50%) + Platform Architecture (30%) + AI Features (20%)

### Stage 7: Visualization & Export

  - Creates the 2D scatter plot (`ecosystem_map.png`).
  - Exports the complete, final dataset to `ecosystem_data_final.xlsx`.

### Scoring Rubrics

#### Quantitative Metrics (X-Axis: Dominance)

  - **Financial Strength (70% weight)**: Total funding, market cap, annual revenue.
  - **Market Penetration (30% weight)**: Number of employees and web traffic.

#### Qualitative Metrics (Y-Axis: AI Ready)

  - **API Quality Score (50% weight)**: Scored 0-3 based on availability and quality of APIs/SDKs.
  - **Platform Architecture Score (30% weight)**: Scored 1-3 based on whether the tech is legacy on-premise vs. modern cloud-native.
  - **Native AI/ML Features (20% weight)**: Scored 0-1 based on the availability of built-in AI capabilities.

-----

## Configuration & Customization

### Adjusting Metric Weights

You can edit the `MetricWeights` class directly in `ecosystem_map.py` to change the importance of each component in the final score.

### Adding New Companies

Edit the `companies.xlsx` file to add or remove companies and re-run the script. A sample file with 17 biotech companies is included.

-----

## Technical Details

  - **Multi-Source Scraping**: The script uses a fallback chain (Google → DuckDuckGo → Perplexity) to increase the chances of successful data collection.
  - **Reliability**: Progress is saved in checkpoints every 5 companies, allowing for automatic recovery from interruptions. The script includes robust error handling and can degrade gracefully if a component (like the OpenAI API) fails.

## Troubleshooting

  - **ChromeDriver Issues**: The script automatically manages ChromeDriver. Ensure the Chrome browser is installed and up-to-date.
  - **OpenAI Errors**: Verify your API key is correct in `.env` and that your OpenAI account has sufficient credits. The script can run without a key by using a fallback scoring mode, but the qualitative analysis will be less accurate.
  - **Package Installation**: If `setup.py` fails, you can install packages manually with `pip install -r requirements.txt`.
