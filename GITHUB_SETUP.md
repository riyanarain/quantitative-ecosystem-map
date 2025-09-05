# GitHub Setup Instructions

This file contains step-by-step instructions for uploading the Quantitative Ecosystem Map Generator to GitHub.

## Prerequisites

1. **Install Git** (if not already installed):
   ```bash
   xcode-select --install
   ```
   Complete the installation dialog that appears.

2. **GitHub Account**: Make sure you have a GitHub account at [github.com](https://github.com)

## Method 1: Command Line (Recommended)

### Step 1: Initialize Local Repository
```bash
cd /Users/rnarain/ecosystem_map
git init
git add .
git commit -m "Initial commit: Quantitative Ecosystem Map Generator"
```

### Step 2: Create GitHub Repository
1. Go to [github.com](https://github.com) and log in
2. Click the "+" icon (top right) → "New repository"
3. Repository settings:
   - **Name**: `quantitative-ecosystem-map`
   - **Description**: `Python script for generating ecosystem maps using web scraping and LLM analysis`
   - **Visibility**: Public (recommended) or Private
   - **Do NOT check**: "Add a README file" (we already have one)
   - **Do NOT check**: "Add .gitignore" (we already have one)
   - **Do NOT check**: "Choose a license"

### Step 3: Connect and Push
Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/quantitative-ecosystem-map.git
git branch -M main
git push -u origin main
```

### Step 4: Verify Upload
Visit your repository URL to see all files uploaded successfully.

## Method 2: GitHub Desktop (GUI Alternative)

1. **Download and Install**: [GitHub Desktop](https://desktop.github.com/)
2. **Sign In**: Log in with your GitHub account
3. **Add Repository**: 
   - Click "Add an Existing Repository from your Hard Drive"
   - Browse to `/Users/rnarain/ecosystem_map`
   - Click "Add Repository"
4. **Initial Commit**:
   - Review files to be committed
   - Add commit message: "Initial commit: Quantitative Ecosystem Map Generator"
   - Click "Commit to main"
5. **Publish**:
   - Click "Publish repository"
   - Set name: `quantitative-ecosystem-map`
   - Add description
   - Choose Public/Private
   - Click "Publish Repository"

## Method 3: VS Code Integration

1. **Open Project**: Open the `/Users/rnarain/ecosystem_map` folder in VS Code
2. **Initialize Git**: 
   - Click Source Control icon (left sidebar)
   - Click "Initialize Repository"
3. **Stage and Commit**:
   - Stage all files (click + next to "Changes")
   - Enter commit message: "Initial commit: Quantitative Ecosystem Map Generator"
   - Click "Commit"
4. **Publish**: Click "Publish to GitHub" and follow prompts

## What Gets Uploaded

The following files will be included in your GitHub repository:

✅ **Core Scripts**:
- `ecosystem_map.py` - Main application
- `demo_with_mock_data.py` - Demo script
- `test_functionality.py` - Test suite

✅ **Setup & Configuration**:
- `setup.py` - Installation script
- `requirements.txt` - Dependencies
- `companies.xlsx` - Sample data

✅ **Documentation**:
- `README.md` - Complete usage guide
- `project_summary.py` - Project overview

✅ **Utilities**:
- `create_sample_data.py` - Data generation utility
- `.gitignore` - Git ignore rules

❌ **Excluded Files** (via .gitignore):
- `.env` - API keys (keep private!)
- Generated output files (`*.png`, `*.xlsx` results)
- Python cache files (`__pycache__/`)
- System files (`.DS_Store`)

## Post-Upload Steps

1. **Update README**: Edit the repository README on GitHub to add:
   - Installation badges
   - Demo images (upload sample outputs)
   - Contribution guidelines

2. **Add Topics**: In your GitHub repo settings, add topics like:
   - `python`
   - `data-visualization`
   - `web-scraping`
   - `openai`
   - `selenium`
   - `ecosystem-analysis`

3. **Create Releases**: Tag important versions using GitHub's release feature

4. **Set up Issues**: Enable issue tracking for bug reports and feature requests

## Security Notes

- **Never commit `.env` files** with real API keys
- The `.gitignore` file prevents this automatically
- If you accidentally commit secrets, immediately:
  1. Remove them from the repository
  2. Regenerate the API keys
  3. Update your local `.env` file

## Repository URL

After creation, your repository will be available at:
```
https://github.com/YOUR_USERNAME/quantitative-ecosystem-map
```

## Need Help?

- **Git documentation**: [git-scm.com](https://git-scm.com/doc)
- **GitHub guides**: [docs.github.com](https://docs.github.com)
- **GitHub Desktop help**: [docs.github.com/desktop](https://docs.github.com/desktop)
