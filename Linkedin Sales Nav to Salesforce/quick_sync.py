#!/usr/bin/env python3
"""
Quick LinkedIn to Salesforce Lead Import Script
This script allows manual CSV import from LinkedIn Sales Navigator exports
"""

import csv
import json
import requests
from datetime import datetime


def parse_linkedin_csv(csv_file):
    """Parse LinkedIn Sales Navigator CSV export"""
    leads = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lead = {
                'name': f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip(),
                'title': row.get('Title', ''),
                'company': row.get('Company', ''),
                'location': row.get('Location', ''),
                'linkedin_url': row.get('LinkedIn Profile', ''),
                'email': row.get('Email', ''),
                'phone': row.get('Phone', '')
            }
            leads.append(lead)
    
    return leads


def sync_to_salesforce_via_dataloader(leads, output_file='salesforce_import.csv'):
    """Format leads for Salesforce Data Loader import"""
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'FirstName', 'LastName', 'Company', 'Title', 
            'Email', 'Phone', 'City', 'LeadSource', 
            'Status', 'Description'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for lead in leads:
            name_parts = lead['name'].split(' ', 1)
            first_name = name_parts[0] if len(name_parts) > 0 else ''
            last_name = name_parts[1] if len(name_parts) > 1 else lead['name']
            
            sf_lead = {
                'FirstName': first_name,
                'LastName': last_name,
                'Company': lead['company'] or 'Unknown',
                'Title': lead['title'],
                'Email': lead['email'],
                'Phone': lead['phone'],
                'City': lead['location'].split(',')[0] if lead['location'] else '',
                'LeadSource': 'LinkedIn Sales Navigator',
                'Status': 'New',
                'Description': f"LinkedIn: {lead['linkedin_url']}"
            }
            
            writer.writerow(sf_lead)
    
    print(f"Created {output_file} with {len(leads)} leads for Salesforce import")


if __name__ == "__main__":
    # Example usage
    linkedin_csv = input("Enter LinkedIn CSV filename: ")
    leads = parse_linkedin_csv(linkedin_csv)
    sync_to_salesforce_via_dataloader(leads)
    print("\nNext steps:")
    print("1. Open Salesforce Data Loader")
    print("2. Choose 'Insert' operation")
    print("3. Select 'Lead' object")
    print("4. Import salesforce_import.csv")
