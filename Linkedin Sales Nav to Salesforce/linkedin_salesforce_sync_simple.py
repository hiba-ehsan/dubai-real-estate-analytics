import os
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SalesforceAPI:
    """Simplified Salesforce API connector using REST API"""
    
    def __init__(self, username: str, password: str, security_token: str, 
                 client_id: str, client_secret: str, domain: str = 'login'):
        self.username = username
        self.password = password + security_token
        self.client_id = client_id
        self.client_secret = client_secret
        self.domain = domain
        self.access_token = None
        self.instance_url = None
        
    def authenticate(self):
        """Authenticate with Salesforce OAuth2"""
        auth_url = f"https://{self.domain}.salesforce.com/services/oauth2/token"
        
        data = {
            'grant_type': 'password',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'username': self.username,
            'password': self.password
        }
        
        try:
            response = requests.post(auth_url, data=data)
            response.raise_for_status()
            
            auth_data = response.json()
            self.access_token = auth_data['access_token']
            self.instance_url = auth_data['instance_url']
            
            logger.info("Successfully authenticated with Salesforce")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def create_lead(self, lead_data: Dict) -> Optional[str]:
        """Create a lead using REST API"""
        if not self.access_token:
            logger.error("Not authenticated")
            return None
        
        url = f"{self.instance_url}/services/data/v58.0/sobjects/Lead"
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        # Map lead data
        sf_lead = {
            'LastName': lead_data.get('name', '').split()[-1] if lead_data.get('name') else 'Unknown',
            'FirstName': ' '.join(lead_data.get('name', '').split()[:-1]) if lead_data.get('name') else '',
            'Company': lead_data.get('company', 'Unknown'),
            'Title': lead_data.get('title', ''),
            'City': lead_data.get('location', '').split(',')[0] if lead_data.get('location') else '',
            'LeadSource': 'LinkedIn Sales Navigator',
            'Status': 'New',
            'Description': f"LinkedIn URL: {lead_data.get('linkedin_url', '')}"
        }
        
        try:
            response = requests.post(url, headers=headers, json=sf_lead)
            
            if response.status_code == 201:
                result = response.json()
                logger.info(f"Created lead: {lead_data.get('name')} - ID: {result.get('id')}")
                return result.get('id')
            else:
                logger.error(f"Failed to create lead: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating lead: {str(e)}")
            return None


def create_quick_sync_script():
    """Create a quick sync script that can run without Chrome driver"""
    script_content = '''#!/usr/bin/env python3
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
    print("\\nNext steps:")
    print("1. Open Salesforce Data Loader")
    print("2. Choose 'Insert' operation")
    print("3. Select 'Lead' object")
    print("4. Import salesforce_import.csv")
'''
    
    with open('quick_sync.py', 'w') as f:
        f.write(script_content)
    
    logger.info("Created quick_sync.py for manual CSV processing")


# Create the quick sync script
create_quick_sync_script()


# Main execution info
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LinkedIn Sales Navigator to Salesforce Sync Setup Complete!")
    print("="*60)
    
    print("\n📁 Files created:")
    print("1. linkedin_salesforce_sync.py - Full automation script")
    print("2. linkedin_salesforce_sync_simple.py - Alternative REST API version")
    print("3. quick_sync.py - Manual CSV processing script")
    print("4. requirements.txt - Required Python packages")
    print("5. setup.py - Package installer script")
    
    print("\n🚀 To get started:")
    print("\n  Option 1: Full Automation (Recommended)")
    print("  1. Install Chrome WebDriver from: https://chromedriver.chromium.org/")
    print("  2. Run: python setup.py")
    print("  3. Run: python linkedin_salesforce_sync.py")
    print("  4. Update the generated config.json with your credentials")
    print("  5. Run again: python linkedin_salesforce_sync.py")
    
    print("\n  Option 2: Manual CSV Import")
    print("  1. Export leads from LinkedIn Sales Navigator as CSV")
    print("  2. Run: python quick_sync.py")
    print("  3. Import the generated CSV using Salesforce Data Loader")
    
    print("\n⚠️  Important Notes:")
    print("- Chrome WebDriver must match your Chrome version")
    print("- You need Salesforce API access (Enterprise/Unlimited edition)")
    print("- For Salesforce authentication, you'll need:")
    print("  - Username and password")
    print("  - Security token (reset from Salesforce settings)")
    print("  - Connected App credentials (for REST API)")
    
    print("\n📋 Configuration needed in config.json:")
    print(json.dumps({
        "linkedin": {
            "email": "your_email@example.com",
            "password": "your_password"
        },
        "salesforce": {
            "username": "your_sf_username@example.com",
            "password": "your_sf_password",
            "security_token": "your_security_token",
            "domain": "login"
        },
        "max_pages": 5,
        "headless": False
    }, indent=2))
