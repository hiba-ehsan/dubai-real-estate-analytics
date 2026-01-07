import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import requests
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('linkedin_salesforce_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LinkedInSalesNavigator:
    """Class to handle LinkedIn Sales Navigator operations"""
    
    def __init__(self, email: str, password: str, headless: bool = False):
        self.email = email
        self.password = password
        self.driver = None
        self.headless = headless
        self.leads_data = []
        
    def setup_driver(self):
        """Set up Chrome driver with options"""
        chrome_options = Options()
        
        # Add Chrome options
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # User agent to avoid detection
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
    def login(self):
        """Login to LinkedIn"""
        try:
            logger.info("Logging into LinkedIn...")
            self.driver.get("https://www.linkedin.com/login")
            time.sleep(2)
            
            # Enter credentials
            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            email_field.send_keys(self.email)
            
            password_field = self.driver.find_element(By.ID, "password")
            password_field.send_keys(self.password)
            
            # Click login button
            login_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            # Wait for login to complete
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, "global-nav"))
            )
            logger.info("Successfully logged into LinkedIn")
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            raise
    
    def navigate_to_sales_navigator(self):
        """Navigate to Sales Navigator"""
        try:
            logger.info("Navigating to Sales Navigator...")
            self.driver.get("https://www.linkedin.com/sales/")
            time.sleep(3)
            
            # Wait for Sales Navigator to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "global-nav__primary-items"))
            )
            logger.info("Successfully navigated to Sales Navigator")
            
        except Exception as e:
            logger.error(f"Failed to navigate to Sales Navigator: {str(e)}")
            raise
    
    def extract_leads(self, max_pages: int = 5):
        """Extract leads from Sales Navigator"""
        try:
            logger.info("Extracting leads from Sales Navigator...")
            
            # Navigate to leads list
            self.driver.get("https://www.linkedin.com/sales/lists/people")
            time.sleep(3)
            
            page_count = 0
            
            while page_count < max_pages:
                # Wait for results to load
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "search-results__result-list"))
                )
                
                # Extract lead information
                leads = self.driver.find_elements(By.CLASS_NAME, "result-lockup__name")
                
                for lead in leads:
                    try:
                        lead_data = self._extract_lead_info(lead)
                        if lead_data:
                            self.leads_data.append(lead_data)
                    except Exception as e:
                        logger.warning(f"Failed to extract lead info: {str(e)}")
                        continue
                
                # Try to go to next page
                try:
                    next_button = self.driver.find_element(By.XPATH, "//button[@aria-label='Next']")
                    if next_button.is_enabled():
                        next_button.click()
                        time.sleep(3)
                        page_count += 1
                    else:
                        break
                except NoSuchElementException:
                    logger.info("No more pages to process")
                    break
            
            logger.info(f"Extracted {len(self.leads_data)} leads")
            return self.leads_data
            
        except Exception as e:
            logger.error(f"Failed to extract leads: {str(e)}")
            raise
    
    def _extract_lead_info(self, lead_element) -> Optional[Dict]:
        """Extract information from a single lead element"""
        try:
            lead_info = {}
            
            # Get lead container
            lead_container = lead_element.find_element(By.XPATH, "./ancestor::li")
            
            # Extract name
            name_element = lead_container.find_element(By.CLASS_NAME, "result-lockup__name")
            lead_info['name'] = name_element.text.strip()
            
            # Extract title
            try:
                title_element = lead_container.find_element(By.CLASS_NAME, "result-lockup__highlight")
                lead_info['title'] = title_element.text.strip()
            except:
                lead_info['title'] = ""
            
            # Extract company
            try:
                company_element = lead_container.find_element(By.CLASS_NAME, "result-lockup__position-company")
                lead_info['company'] = company_element.text.strip()
            except:
                lead_info['company'] = ""
            
            # Extract location
            try:
                location_element = lead_container.find_element(By.CLASS_NAME, "result-lockup__misc-item")
                lead_info['location'] = location_element.text.strip()
            except:
                lead_info['location'] = ""
            
            # Extract profile URL
            try:
                profile_link = lead_container.find_element(By.CLASS_NAME, "result-lockup__icon-link")
                lead_info['linkedin_url'] = profile_link.get_attribute('href')
            except:
                lead_info['linkedin_url'] = ""
            
            # Add timestamp
            lead_info['extracted_date'] = datetime.now().isoformat()
            
            return lead_info
            
        except Exception as e:
            logger.warning(f"Failed to extract lead info: {str(e)}")
            return None
    
    def save_leads_to_csv(self, filename: str = 'linkedin_leads.csv'):
        """Save extracted leads to CSV file"""
        if not self.leads_data:
            logger.warning("No leads data to save")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['name', 'title', 'company', 'location', 'linkedin_url', 'extracted_date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for lead in self.leads_data:
                writer.writerow(lead)
        
        logger.info(f"Saved {len(self.leads_data)} leads to {filename}")
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()


class SalesforceConnector:
    """Class to handle Salesforce operations"""
    
    def __init__(self, username: str, password: str, security_token: str, domain: str = 'login'):
        self.username = username
        self.password = password
        self.security_token = security_token
        self.domain = domain
        self.sf = None
        
    def connect(self):
        """Connect to Salesforce using simple-salesforce"""
        try:
            from simple_salesforce import Salesforce
            
            logger.info("Connecting to Salesforce...")
            self.sf = Salesforce(
                username=self.username,
                password=self.password,
                security_token=self.security_token,
                domain=self.domain
            )
            logger.info("Successfully connected to Salesforce")
            
        except ImportError:
            logger.error("simple-salesforce package not installed. Please run: pip install simple-salesforce")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Salesforce: {str(e)}")
            raise
    
    def create_lead(self, lead_data: Dict) -> Optional[str]:
        """Create a lead in Salesforce"""
        try:
            # Map LinkedIn data to Salesforce Lead fields
            sf_lead = {
                'LastName': lead_data.get('name', '').split()[-1] if lead_data.get('name') else 'Unknown',
                'FirstName': ' '.join(lead_data.get('name', '').split()[:-1]) if lead_data.get('name') else '',
                'Company': lead_data.get('company', 'Unknown'),
                'Title': lead_data.get('title', ''),
                'City': lead_data.get('location', '').split(',')[0] if lead_data.get('location') else '',
                'LinkedIn_URL__c': lead_data.get('linkedin_url', ''),  # Custom field
                'LeadSource': 'LinkedIn Sales Navigator',
                'Status': 'New'
            }
            
            # Create lead in Salesforce
            result = self.sf.Lead.create(sf_lead)
            
            if result.get('success'):
                logger.info(f"Created lead: {lead_data.get('name')} - ID: {result.get('id')}")
                return result.get('id')
            else:
                logger.error(f"Failed to create lead: {lead_data.get('name')}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating lead in Salesforce: {str(e)}")
            return None
    
    def check_existing_lead(self, email: str = None, linkedin_url: str = None) -> bool:
        """Check if lead already exists in Salesforce"""
        try:
            query = "SELECT Id, Name FROM Lead WHERE "
            
            if email:
                query += f"Email = '{email}'"
            elif linkedin_url:
                query += f"LinkedIn_URL__c = '{linkedin_url}'"
            else:
                return False
            
            result = self.sf.query(query)
            
            return result['totalSize'] > 0
            
        except Exception as e:
            logger.warning(f"Error checking existing lead: {str(e)}")
            return False
    
    def bulk_create_leads(self, leads_data: List[Dict]) -> Dict:
        """Bulk create leads in Salesforce"""
        results = {
            'created': 0,
            'skipped': 0,
            'failed': 0
        }
        
        for lead_data in leads_data:
            # Check if lead already exists
            if self.check_existing_lead(linkedin_url=lead_data.get('linkedin_url')):
                logger.info(f"Lead already exists: {lead_data.get('name')}")
                results['skipped'] += 1
                continue
            
            # Create lead
            lead_id = self.create_lead(lead_data)
            if lead_id:
                results['created'] += 1
            else:
                results['failed'] += 1
        
        return results


class LinkedInSalesforceSync:
    """Main class to orchestrate the sync between LinkedIn and Salesforce"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.linkedin = None
        self.salesforce = None
        
    def run_sync(self):
        """Run the complete sync process"""
        try:
            # Initialize LinkedIn Sales Navigator
            logger.info("Starting LinkedIn Sales Navigator sync...")
            self.linkedin = LinkedInSalesNavigator(
                email=self.config['linkedin']['email'],
                password=self.config['linkedin']['password'],
                headless=self.config.get('headless', False)
            )
            
            # Setup and login to LinkedIn
            self.linkedin.setup_driver()
            self.linkedin.login()
            self.linkedin.navigate_to_sales_navigator()
            
            # Extract leads
            leads_data = self.linkedin.extract_leads(max_pages=self.config.get('max_pages', 5))
            
            # Save to CSV as backup
            self.linkedin.save_leads_to_csv()
            
            # Initialize Salesforce connection
            logger.info("Connecting to Salesforce...")
            self.salesforce = SalesforceConnector(
                username=self.config['salesforce']['username'],
                password=self.config['salesforce']['password'],
                security_token=self.config['salesforce']['security_token'],
                domain=self.config['salesforce'].get('domain', 'login')
            )
            self.salesforce.connect()
            
            # Sync leads to Salesforce
            logger.info("Syncing leads to Salesforce...")
            results = self.salesforce.bulk_create_leads(leads_data)
            
            logger.info(f"Sync completed: Created: {results['created']}, Skipped: {results['skipped']}, Failed: {results['failed']}")
            
        except Exception as e:
            logger.error(f"Sync failed: {str(e)}")
            raise
        finally:
            if self.linkedin:
                self.linkedin.close()


def load_config(config_file: str = 'config.json') -> Dict:
    """Load configuration from file"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Return template config
        return {
            "linkedin": {
                "email": "your_linkedin_email@example.com",
                "password": "your_linkedin_password"
            },
            "salesforce": {
                "username": "your_salesforce_username@example.com",
                "password": "your_salesforce_password",
                "security_token": "your_salesforce_security_token",
                "domain": "login"  # or 'test' for sandbox
            },
            "max_pages": 5,
            "headless": False
        }


def main():
    """Main execution function"""
    # Load configuration
    config = load_config()
    
    # Check if config needs to be updated
    if config['linkedin']['email'] == 'your_linkedin_email@example.com':
        logger.error("Please update the config.json file with your credentials")
        
        # Save template config
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\nTemplate config.json created. Please update it with your credentials:")
        print("1. LinkedIn email and password")
        print("2. Salesforce username, password, and security token")
        print("3. Set 'domain' to 'test' if using Salesforce sandbox")
        return
    
    # Run sync
    sync = LinkedInSalesforceSync(config)
    sync.run_sync()


if __name__ == "__main__":
    main()
