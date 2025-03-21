import requests
import pandas as pd
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import logging
import os

class ScryfallFetcher:
    """
    A class to handle fetching and processing Magic: The Gathering cards from Scryfall API
    with proper rate limiting and header handling
    """
    
    BASE_URL = "https://api.scryfall.com"
    
    def __init__(self, app_name: str = "MTGDeckBuilder/1.0"):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': app_name,
            'Accept': 'application/json'
        })
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, 
                     format_type: str = 'json') -> Dict:
        """
        Make a request to Scryfall API with proper rate limiting and error handling
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            format_type: Response format (json, csv, text, image)
        """
        if params is None:
            params = {}
            
        if format_type != 'json':
            params['format'] = format_type
            
        # Rate limiting - 100ms between requests as per documentation
        time.sleep(0.1)
        
        try:
            response = self.session.get(f"{self.BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            
            # Handle different response formats
            if format_type == 'json':
                return response.json()
            elif format_type == 'csv':
                return response.text
            else:
                return response.content
                
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                self.logger.error("Rate limit exceeded. Waiting before retry...")
                time.sleep(1)  # Wait longer before retry
                return self._make_request(endpoint, params, format_type)
            else:
                self.logger.error(f"API request failed: {str(e)}")
                raise
    
    def get_standard_cards(self) -> List[Dict[str, Any]]:
        """
        Fetch all Standard-legal cards with proper pagination handling
        """
        cards = []
        params = {
            'q': 'format:standard legal:standard',
            'unique': 'cards'
        }
        
        self.logger.info("Starting to fetch Standard-legal cards...")
        
        try:
            # Initial request
            response = self._make_request('cards/search', params)
            cards.extend(response.get('data', []))
            total_cards = response.get('total_cards', 0)
            
            # Handle pagination
            while response.get('has_more'):
                self.logger.info(f"Fetched {len(cards)}/{total_cards} cards...")
                response = self._make_request(response['next_page'].replace(f"{self.BASE_URL}/", ''))
                cards.extend(response.get('data', []))
            
            self.logger.info(f"Completed fetching {len(cards)} cards")
            return cards
            
        except Exception as e:
            self.logger.error(f"Error fetching cards: {str(e)}")
            raise
    
    def process_card_data(self, cards: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process raw card data into a structured DataFrame with enhanced handling of special layouts
        """
        processed_cards = []
        
        for card in cards:
            # Handle different layouts (split, adventure, transform, modal_dfc, etc.)
            card_name = card.get('name')
            layout = card.get('layout', '')
            
            # For split cards and other multi-face cards
            if '//' in card_name or layout in ['split', 'adventure', 'modal_dfc', 'transform']:
                # Store both the full name and the front face name
                if '//' in card_name:
                    front_name = card_name.split('//')[0].strip()
                else:
                    front_name = card_name
                
                # Initialize variables for card faces
                front_face_oracle = ""
                back_face_oracle = ""
                front_face_type = ""
                back_face_type = ""
                mana_cost = ""
                colors = []
                
                # Get face-specific data from card_faces
                if 'card_faces' in card and card['card_faces']:
                    front_face = card['card_faces'][0]
                    mana_cost = front_face.get('mana_cost', card.get('mana_cost', ''))
                    colors = front_face.get('colors', card.get('colors', []))
                    front_face_oracle = front_face.get('oracle_text', '')
                    front_face_type = front_face.get('type_line', '')
                    
                    # Process back face if it exists
                    if len(card['card_faces']) > 1:
                        back_face = card['card_faces'][1]
                        back_face_oracle = back_face.get('oracle_text', '')
                        back_face_type = back_face.get('type_line', '')
                        
                        # For Room cards, we want to combine the oracle text to capture both abilities
                        if "Room" in front_face_type and "Room" in back_face_type:
                            combined_oracle = f"{front_face_oracle}\n\n{back_face_oracle}"
                        else:
                            combined_oracle = front_face_oracle
                else:
                    # Fallback if card_faces is not available
                    mana_cost = card.get('mana_cost', '')
                    colors = card.get('colors', [])
                    front_face_oracle = card.get('oracle_text', '')
                    front_face_type = card.get('type_line', '')
                    combined_oracle = front_face_oracle
                
                # For type_line, we want to capture both sides for split cards
                if back_face_type and front_face_type:
                    combined_type_line = f"{front_face_type} // {back_face_type}"
                else:
                    combined_type_line = card.get('type_line', '')
                
                # Store both names for reference
                processed_card = {
                    'name': front_name,  # Store front face name as primary name
                    'full_name': card_name,  # Store full split name
                    'layout': layout,
                    'mana_cost': mana_cost,
                    'cmc': card.get('cmc'),
                    'type_line': combined_type_line,
                    'oracle_text': combined_oracle,
                    'colors': colors,
                    'color_identity': card.get('color_identity', []),
                    'power': front_face.get('power', card.get('power', '')),
                    'toughness': front_face.get('toughness', card.get('toughness', '')),
                    'rarity': card.get('rarity'),
                    'set': card.get('set'),
                    'collector_number': card.get('collector_number'),
                    'keywords': card.get('keywords', []),
                    'produced_mana': card.get('produced_mana', []),
                    'legalities': card.get('legalities', {}),
                }
            else:
                # Normal card processing
                processed_card = {
                    'name': card_name,
                    'full_name': card_name,
                    'layout': layout,
                    'mana_cost': card.get('mana_cost'),
                    'cmc': card.get('cmc'),
                    'type_line': card.get('type_line'),
                    'oracle_text': card.get('oracle_text'),
                    'colors': card.get('colors', []),
                    'color_identity': card.get('color_identity', []),
                    'power': card.get('power'),
                    'toughness': card.get('toughness'),
                    'rarity': card.get('rarity'),
                    'set': card.get('set'),
                    'collector_number': card.get('collector_number'),
                    'keywords': card.get('keywords', []),
                    'produced_mana': card.get('produced_mana', []),
                    'legalities': card.get('legalities', {}),
                }
            
            # Add derived features
            processed_card.update({
                'is_creature': 'Creature' in processed_card['type_line'],
                'is_land': 'Land' in processed_card['type_line'],
                'is_instant_sorcery': any(t in processed_card['type_line'] 
                                        for t in ['Instant', 'Sorcery']),
                'is_multicolored': len(processed_card['colors']) > 1,
                'color_count': len(processed_card['colors']),
                'has_etb_effect': 'enters the battlefield' in (processed_card['oracle_text'] or '').lower(),
                'is_legendary': 'Legendary' in processed_card['type_line']
            })
            
            processed_cards.append(processed_card)
            
        return pd.DataFrame(processed_cards)

    def save_to_csv(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save the processed cards to a CSV file in the data directory
        """
        # Create data directory if it doesn't exist
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        
        if filename is None:
            filename = f"standard_cards.csv"
        
        # Construct full path
        filepath = os.path.join(data_dir, filename)
        
        df.to_csv(filepath, index=False)
        return filepath

def main():
    """
    Main function to demonstrate usage
    """
    fetcher = ScryfallFetcher()
    
    try:
        print("Fetching Standard-legal cards...")
        cards = fetcher.get_standard_cards()
        
        print("Processing card data...")
        df = fetcher.process_card_data(cards)
        
        filename = fetcher.save_to_csv(df)
        print(f"Data saved to {filename}")
        
        print("\nDataset Overview:")
        print(f"Total cards: {len(df)}")
        print("\nCard type distribution:")
        print(f"Creatures: {df['is_creature'].sum()}")
        print(f"Lands: {df['is_land'].sum()}")
        print(f"Instants/Sorceries: {df['is_instant_sorcery'].sum()}")
        print(f"Legendary cards: {df['is_legendary'].sum()}")
        
        print("\nColor distribution:")
        print(f"Multicolored cards: {df['is_multicolored'].sum()}")
        color_counts = df['color_count'].value_counts().sort_index()
        for count, num_cards in color_counts.items():
            print(f"{count} color(s): {num_cards} cards")
        
        # Print a few special layout types to verify
        print("\nSpecial layouts:")
        room_cards = df[df['type_line'].str.contains('Room', na=False)]
        split_cards = df[df['full_name'].str.contains(' // ', na=False)]
        print(f"Room cards: {len(room_cards)}")
        print(f"Split cards: {len(split_cards)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()