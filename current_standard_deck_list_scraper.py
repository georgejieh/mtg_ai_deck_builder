import os
import re
import time
import json
import shutil
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MTGGoldfishScraper:
    def __init__(self, output_dir="current_standard_decks", min_meta_percentage=0.5):
        self.base_url = "https://www.mtggoldfish.com"
        self.metagame_url = f"{self.base_url}/metagame/standard/full#paper"
        self.output_dir = output_dir
        self.min_meta_percentage = min_meta_percentage
        self.session = requests.Session()
        # Setting headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session.headers.update(self.headers)
        # List to keep track of downloaded decks for the JSON export
        self.meta_data = []
        
    def clear_output_directory(self):
        """Clear the output directory to remove old deck lists."""
        if os.path.exists(self.output_dir):
            logger.info(f"Clearing existing directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        # Create fresh directory
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Created fresh directory: {self.output_dir}")
    
    def collect_meta_information(self):
        """
        Collect meta information from the metagame page first
        without downloading any decks yet.
        """
        logger.info(f"Fetching metagame page: {self.metagame_url}")
        response = self.session.get(self.metagame_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        deck_tiles = soup.select('.archetype-tile')
        
        meta_decks = []
        for tile in deck_tiles:
            try:
                # Extract meta percentage
                meta_stat = tile.select_one('.archetype-tile-statistic-value')
                if not meta_stat:
                    continue
                    
                meta_text = meta_stat.get_text(strip=True)
                meta_percentage_match = re.search(r'(\d+\.\d+)%', meta_text)
                if not meta_percentage_match:
                    continue
                    
                meta_percentage = float(meta_percentage_match.group(1))
                
                # Extract number of decks
                extra_data = tile.select_one('.archetype-tile-statistic-value-extra-data')
                deck_count = None
                if extra_data:
                    deck_count_match = re.search(r'\((\d+)\)', extra_data.get_text(strip=True))
                    if deck_count_match:
                        deck_count = int(deck_count_match.group(1))
                
                # Find the archetype link SPECIFICALLY in the title section, not anywhere in the tile
                # This is crucial to avoid grabbing card names instead of the archetype name
                deck_link = tile.select_one('.archetype-tile-title a')
                if not deck_link:
                    continue
                    
                # Get the URL of the deck page
                deck_url = urljoin(self.base_url, deck_link['href'])
                
                # Get the EXACT archetype name from the link text
                archetype_name = deck_link.get_text(strip=True)
                
                # Debug output to verify we're getting the right name
                logger.info(f"Found archetype: '{archetype_name}' at {deck_url}")
                
                # Only collect decks that meet the threshold
                if meta_percentage >= self.min_meta_percentage:
                    meta_decks.append({
                        'archetype_name': archetype_name,
                        'url': deck_url,
                        'meta_percentage': meta_percentage,
                        'deck_count': deck_count
                    })
                    logger.info(f"Added deck: {archetype_name} ({meta_percentage}%, {deck_count} decks)")
            except Exception as e:
                logger.error(f"Error processing a deck tile: {e}")
        
        return meta_decks
    
    def process_deck_page(self, deck_info):
        """
        Process an individual deck page to find the download link.
        """
        archetype_name = deck_info['archetype_name']
        deck_url = deck_info['url']
        
        logger.info(f"Processing deck page for {archetype_name}: {deck_url}")
        
        try:
            # Get the deck page
            response = self.session.get(deck_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the download link
            download_link = soup.select_one('a.dropdown-item[href*="/deck/download/"]')
            if not download_link:
                logger.warning(f"No download link found for {archetype_name}")
                return False
                
            download_url = urljoin(self.base_url, download_link['href'])
            
            # Return the download URL for the next step
            return download_url
        except Exception as e:
            logger.error(f"Error processing deck page for {archetype_name}: {e}")
            return None
    
    def download_and_save_deck(self, download_url, deck_info):
        """
        Download the deck and save it with the correct archetype name.
        """
        archetype_name = deck_info['archetype_name']
        
        try:
            logger.info(f"Downloading deck list for {archetype_name} from: {download_url}")
            
            # Download the deck list content
            response = self.session.get(download_url)
            response.raise_for_status()
            deck_content = response.text
            
            # Create the filename with the archetype name - this is crucial
            filename = f"Deck - {archetype_name}.txt"
            file_path = os.path.join(self.output_dir, filename)
            
            # Save the deck list content directly to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(deck_content)
            
            logger.info(f"Successfully saved deck list to: {file_path}")
            
            # Add the deck to our meta data collection
            self.meta_data.append({
                "archetype": archetype_name,
                "meta_percentage": deck_info['meta_percentage'],
                "deck_count": deck_info.get('deck_count', 0),
                "url": deck_info['url']
            })
            
            return True
        except Exception as e:
            logger.error(f"Error downloading or saving deck for {archetype_name}: {e}")
            return False
    
    def export_meta_json(self):
        """Export the meta data to a JSON file in the json_outputs directory."""
        # Create json_outputs directory if it doesn't exist
        json_dir = "json_outputs"
        os.makedirs(json_dir, exist_ok=True)
    
        json_path = os.path.join(json_dir, "deck_meta_representation.json")
    
        # Sort decks by meta percentage (descending)
        sorted_data = sorted(self.meta_data, key=lambda x: x["meta_percentage"], reverse=True)
    
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=2)
        
        logger.info(f"Exported meta data to: {json_path}")
        return json_path
    
    def run(self):
        """Main execution method with highly separated concerns."""
        try:
            # Step 1: Clear the output directory
            self.clear_output_directory()
            
            # Step 2: Collect meta information without downloading decks
            meta_decks = self.collect_meta_information()
            logger.info(f"Collected information for {len(meta_decks)} decks")
            
            # Step 3: Process each deck sequentially
            successful_downloads = 0
            
            for deck_info in meta_decks:
                try:
                    # Step 3a: Get the download URL from the deck page
                    download_url = self.process_deck_page(deck_info)
                    if not download_url:
                        continue
                    
                    # Step 3b: Download and save with the correct archetype name
                    time.sleep(2)  # Delay to be nice to the server
                    success = self.download_and_save_deck(download_url, deck_info)
                    
                    if success:
                        successful_downloads += 1
                except Exception as e:
                    logger.error(f"Error processing {deck_info['archetype_name']}: {e}")
            
            # Step 4: Export meta data if we have any successful downloads
            if successful_downloads > 0:
                self.export_meta_json()
                
            logger.info(f"Successfully downloaded {successful_downloads} out of {len(meta_decks)} deck lists")
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    scraper = MTGGoldfishScraper()
    scraper.run()