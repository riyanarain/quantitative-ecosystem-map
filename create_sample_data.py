#!/usr/bin/env python3
"""
Create sample Excel file with company names for testing the ecosystem map generator.
"""

import pandas as pd

def create_sample_companies():
    """Create a sample Excel file with laboratory informatics and scientific software companies"""
    
    companies = [
        "Benchling",
        "BenchSci",
        "Biovia",
        "CDD Vault",
        "Certara",
        "Dotmatics",
        "Genedata",
        "Ginkgo Bioworks",
        "KNIME",
        "Labguru",
        "LabWare",
        "LiveDesign (Schrödinger)",
        "ReSync",
        "Revvity",
        "Sapio",
        "SciNote",
        "Scispot",
        "SimulationsPlus",
        "tetrascience",
        "Thermo Scientific"
    ]
    
    # Create DataFrame
    df = pd.DataFrame({'CompanyName': companies})
    
    # Save to Excel
    df.to_excel('companies.xlsx', index=False)
    print(f"Created companies.xlsx with {len(companies)} companies")
    
    # Display the companies
    print("\nCompanies in the dataset:")
    for i, company in enumerate(companies, 1):
        print(f"{i:2d}. {company}")

if __name__ == "__main__":
    create_sample_companies()
