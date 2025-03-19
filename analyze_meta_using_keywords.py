import os
import re
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import Counter, defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_card_name(card_name: str) -> str:
    """
    Normalize card name by standardizing single slash to double slash format
    """
    if not card_name:
        return ""
        
    # Check if the card name contains a single slash but not double slash
    if '/' in card_name and '//' not in card_name:
        # Split by the single slash and rejoin with proper format
        parts = card_name.split('/')
        if len(parts) == 2:
            return f"{parts[0].strip()} // {parts[1].strip()}"
    return card_name

def safe_eval_list(val: Any) -> List:
    """Safely evaluate string representations of lists"""
    if pd.isna(val):
        return []
        
    try:
        if isinstance(val, str):
            # Handle string representation of lists
            if val.startswith('[') and val.endswith(']'):
                # Remove brackets and quotes
                inner = val[1:-1]
                if not inner.strip():
                    return []
                    
                # Split by comma and clean items
                items = []
                for item in inner.split(','):
                    # Clean up quotes and whitespace
                    cleaned = item.strip().strip('\'"')
                    if cleaned:
                        items.append(cleaned)
                return items
            # Handle single values
            elif val.strip():
                return [val.strip()]
            return []
        elif isinstance(val, list):
            return val
        return []
    except Exception as e:
        logger.warning(f"Error parsing list value: {val}. Error: {e}")
        return []

class MetaCardAnalyzer:
    """
    Analyzes card characteristics in the Magic: The Gathering meta.
    Focuses on counting keywords, card types, subtypes, and references in oracle text.
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.unique_keywords = self._extract_unique_keywords()
        self.unique_subtypes = self._extract_unique_subtypes()
        logger.info(f"Meta Card Analyzer initialized with {len(self.unique_keywords)} keywords and {len(self.unique_subtypes)} subtypes")
    
    def _extract_unique_keywords(self) -> Set[str]:
        """Extract all unique keywords from the card database"""
        unique_keywords = set()
        
        for keywords in self.card_db['keywords']:
            if isinstance(keywords, list):
                for keyword in keywords:
                    # Normalize keyword (lowercase)
                    normalized = keyword.lower()
                    if normalized:
                        unique_keywords.add(normalized)
        
        return unique_keywords
    
    def _extract_unique_subtypes(self) -> Set[str]:
        """Extract all unique subtypes from card type lines"""
        unique_subtypes = set()
        
        for type_line in self.card_db['type_line']:
            if not pd.isna(type_line) and '—' in type_line:
                # Split by the em dash to get subtypes
                parts = type_line.split('—', 1)
                if len(parts) > 1:
                    # Get subtypes (after the dash)
                    subtypes_part = parts[1].strip()
                    # Split by spaces and clean
                    for subtype in subtypes_part.split():
                        # Remove any punctuation and normalize
                        subtype = re.sub(r'[^\w\s]', '', subtype).strip().lower()
                        if subtype and subtype not in ['', '/']:  # Exclude empty strings and delimiters
                            unique_subtypes.add(subtype)
        
        return unique_subtypes
    
    def analyze_meta_cards(self, meta_cards_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """
        Analyze all cards in the meta and count various card characteristics.
        
        Returns:
            Dictionary with counts for keywords, card types, subtypes, and references
        """
        # Initialize result dictionaries
        result = {
            'keywords': {},      # Count of cards with each keyword
            'card_types': {},    # Count of each card type
            'subtypes': {},      # Count of each subtype
            'references': {}     # Count of cards referencing types/subtypes
        }
        
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        meta_cards = meta_cards_df.copy()
        
        # Ensure boolean columns are treated as actual booleans
        for col in ['is_creature', 'is_land', 'is_instant_sorcery', 'is_multicolored', 'has_etb_effect', 'is_legendary']:
            if col in meta_cards.columns:
                # Convert string representations to actual boolean values
                meta_cards.loc[:, col] = meta_cards[col].map({'True': True, 'False': False, True: True, False: False})
        
        # Count card types (using properly converted booleans)
        result['card_types']['creature'] = meta_cards['is_creature'].sum()
        result['card_types']['land'] = meta_cards['is_land'].sum()
        result['card_types']['instant_sorcery'] = meta_cards['is_instant_sorcery'].sum()
        result['card_types']['multicolored'] = meta_cards['is_multicolored'].sum()
        result['card_types']['etb_effects'] = meta_cards['has_etb_effect'].sum()
        result['card_types']['legendary'] = meta_cards['is_legendary'].sum()
        
        # Process each card for keywords and subtypes
        for _, card in meta_cards.iterrows():
            self._process_card_keywords(card, result)
            self._process_card_subtypes(card, result)
            self._process_card_references(card, result)
        
        return result
    
    def _process_card_keywords(self, card: pd.Series, result: Dict[str, Dict[str, int]]):
        """Count keywords for a single card (from both keywords column and oracle text)"""
        card_keywords = set()
        
        # Get keywords from the keywords column
        if isinstance(card['keywords'], list):
            for keyword in card['keywords']:
                card_keywords.add(keyword.lower())
        
        # Check oracle text for keywords
        if pd.notna(card['oracle_text']):
            oracle_text = card['oracle_text'].lower()
            
            # Check for each unique keyword in oracle text
            for keyword in self.unique_keywords:
                # Only count if the keyword is fully contained as a word (not part of another word)
                if re.search(r'\b' + re.escape(keyword) + r'\b', oracle_text) and keyword not in card_keywords:
                    card_keywords.add(keyword)
        
        # Update counts
        for keyword in card_keywords:
            if keyword in result['keywords']:
                result['keywords'][keyword] += 1
            else:
                result['keywords'][keyword] = 1
    
    def _process_card_subtypes(self, card: pd.Series, result: Dict[str, Dict[str, int]]):
        """Count subtypes for a single card"""
        if pd.isna(card['type_line']) or '—' not in card['type_line']:
            return
            
        # Extract subtypes
        parts = card['type_line'].split('—', 1)
        if len(parts) > 1:
            subtypes_part = parts[1].strip()
            
            # Split by spaces and clean each subtype
            for subtype in subtypes_part.split():
                # Remove any punctuation and normalize
                subtype = re.sub(r'[^\w\s]', '', subtype).strip().lower()
                if subtype and subtype not in ['', '/']: # Exclude empty strings and delimiters
                    if subtype in result['subtypes']:
                        result['subtypes'][subtype] += 1
                    else:
                        result['subtypes'][subtype] = 1
    
    def _process_card_references(self, card: pd.Series, result: Dict[str, Dict[str, int]]):
        """Count references to card types and subtypes in oracle text"""
        if pd.isna(card['oracle_text']):
            return
            
        oracle_text = card['oracle_text'].lower()
        
        # Check for references to card types
        type_references = {
            'creature_reference': 'creature',
            'land_reference': 'land',
            'instant_reference': 'instant',
            'sorcery_reference': 'sorcery',
            'legendary_reference': 'legendary',
            'etb_reference': 'enters the battlefield'
        }
        
        for ref_key, text in type_references.items():
            # Only count if the reference appears as a word boundary
            if re.search(r'\b' + re.escape(text) + r'\b', oracle_text):
                if ref_key in result['references']:
                    result['references'][ref_key] += 1
                else:
                    result['references'][ref_key] = 1
        
        # Check for references to subtypes
        for subtype in self.unique_subtypes:
            # Only count if the subtype appears as a word boundary
            if re.search(r'\b' + re.escape(subtype) + r'\b', oracle_text):
                ref_key = f"{subtype}_reference"
                if ref_key in result['references']:
                    result['references'][ref_key] += 1
                else:
                    result['references'][ref_key] = 1

class MetaAnalyzer:
    """
    Main meta analysis system that processes decklists and identifies card usage patterns.
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        # Create indices for faster lookups
        self._create_indices()
        # Initialize card analyzer
        self.card_analyzer = MetaCardAnalyzer(card_db)
        logger.info("Meta Analyzer initialized with Card Analyzer")
    
    def _create_indices(self):
        """Create indices for efficient card lookups"""
        # Name-based indices
        self.name_to_card = dict(zip(self.card_db['name'], self.card_db.to_dict('records')))
        self.name_lower_to_card = dict(zip(self.card_db['name'].str.lower(), self.card_db.to_dict('records')))
        
        # Create basic land set for filtering
        self.basic_lands = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest'}
        
        # Create index for full_name to help with dual-faced cards
        self.full_name_to_card = {}
        for _, card in self.card_db.iterrows():
            if pd.notna(card['full_name']):
                self.full_name_to_card[card['full_name']] = card.to_dict()
        
        # Create mapping from front faces to full_names for split cards
        self.front_face_to_full_name = {}
        for full_name, card in self.full_name_to_card.items():
            if ' // ' in full_name:
                front_face = full_name.split(' // ')[0]
                self.front_face_to_full_name[front_face] = full_name
    
    def analyze_meta(self, decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """Perform meta analysis on a collection of decklists"""
        try:
            # Convert decklists to card objects for efficient processing
            processed_decklists = self._process_decklists(decklists)
            
            # Calculate meta speed
            meta_speed = self._calculate_meta_speed(processed_decklists)
            logger.info(f"Meta speed calculated as {meta_speed['speed']}")
            
            # Analyze format characteristics (using our new approach)
            format_characteristics = self._analyze_format_characteristics(processed_decklists)
            logger.info("Format characteristics analyzed")
            
            # Collect card frequencies
            card_frequencies = Counter()
            for deck_info in processed_decklists.values():
                for card, count in deck_info['card_counts'].items():
                    if card not in self.basic_lands:
                        card_frequencies[card] += count
            
            # Calculate meta statistics
            meta_statistics = self._calculate_meta_statistics(meta_speed, card_frequencies, len(processed_decklists))
            
            logger.info("Meta analysis completed successfully")
            return {
                'meta_speed': meta_speed,
                'format_characteristics': format_characteristics,
                'meta_statistics': meta_statistics,
                'card_frequencies': dict(card_frequencies)
            }
            
        except Exception as e:
            logger.error(f"Error in meta analysis: {str(e)}")
            raise
    
    def _process_decklists(self, decklists: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Process decklists into a more efficient format"""
        processed = {}
        for deck_name, card_list in decklists.items():
            # Normalize card names in the decklist (convert single slash to double slash)
            normalized_card_list = [normalize_card_name(card) for card in card_list]
            
            # Count cards
            card_counts = Counter(normalized_card_list)
            
            # Match cards in database
            matched_cards_df = self._match_cards_in_database(card_counts.keys())
            
            # Calculate missing cards by comparing matched card names
            missing_cards = set(card_counts.keys()) - set(matched_cards_df['name'])
            
            processed[deck_name] = {
                'card_counts': card_counts,
                'cards_df': matched_cards_df,
                'missing_cards': missing_cards
            }
            
            if missing_cards:
                logger.warning(f"Deck {deck_name} has {len(missing_cards)} unrecognized cards: {missing_cards}")
        
        return processed
    
    def _match_cards_in_database(self, card_names: set) -> pd.DataFrame:
        """Match card names to database entries with handling of dual-faced cards"""
        # Create an empty DataFrame to store matched cards
        matched_cards = pd.DataFrame()
        remaining_cards = set(card_names)
        
        # Try exact name match
        exact_matches = self.card_db[self.card_db['name'].isin(remaining_cards)]
        if not exact_matches.empty:
            matched_cards = pd.concat([matched_cards, exact_matches])
            remaining_cards -= set(exact_matches['name'])
            
        # For each remaining card, try to match with full_name
        if remaining_cards:
            full_name_matches = self.card_db[self.card_db['full_name'].isin(remaining_cards)]
            if not full_name_matches.empty:
                matched_cards = pd.concat([matched_cards, full_name_matches])
                remaining_cards -= set(full_name_matches['full_name'])
                
        # For each remaining card, check if it might be a split card
        if remaining_cards:
            for card_name in list(remaining_cards):
                # Check if this is a split card with '//' format
                if '//' in card_name:
                    front_face = card_name.split(' // ')[0].strip()
                    
                    # Try to find the front face in the database
                    front_face_matches = self.card_db[self.card_db['name'] == front_face]
                    if not front_face_matches.empty:
                        matched_cards = pd.concat([matched_cards, front_face_matches])
                        remaining_cards.remove(card_name)
        
        # Return all matched cards
        return matched_cards
    
    def _calculate_meta_speed(self, processed_decklists: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate meta speed characteristics without hardcoded interaction check"""
        total_cmc = 0
        total_cards = 0
        early_game_count = 0
        
        for deck_info in processed_decklists.values():
            cards_df = deck_info['cards_df']
            
            # Skip if no cards matched
            if cards_df.empty:
                continue
                
            # Count cards and their CMC
            for _, card in cards_df.iterrows():
                card_name = card['name']
                count = deck_info['card_counts'].get(card_name, 0)
                
                if count > 0 and not card['is_land']:
                    total_cards += count
                    total_cmc += card['cmc'] * count
                    
                    # Count early game cards (CMC <= 2)
                    if card['cmc'] <= 2:
                        early_game_count += count
        
        # Calculate average values
        if total_cards == 0:
            return {
                'speed': 'unknown',
                'avg_cmc': 0.0,
                'early_game_ratio': 0.0,
                'interaction_ratio': 0.0  # Keeping this for compatibility
            }
        
        avg_cmc = total_cmc / total_cards
        early_game_ratio = early_game_count / total_cards
        
        # Determine meta speed
        if avg_cmc < 2.5 and early_game_ratio > 0.3:
            speed = 'fast'
        elif avg_cmc > 3.5 or early_game_ratio < 0.2:
            speed = 'medium'
        else:
            speed = 'slow'
        
        return {
            'speed': speed,
            'avg_cmc': avg_cmc,
            'early_game_ratio': early_game_ratio,
            'interaction_ratio': 0.0  # Placeholder value for compatibility
        }
    
    def _analyze_format_characteristics(self, processed_decklists: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze format characteristics using our new approach"""
        # Get all cards used in all decklists
        all_meta_cards = set()
        for deck_info in processed_decklists.values():
            cards_df = deck_info['cards_df']
            all_meta_cards.update(cards_df['name'])
        
        # Create a DataFrame with all the meta cards
        meta_cards_df = self.card_db[self.card_db['name'].isin(all_meta_cards)]
        
        # Use the MetaCardAnalyzer to analyze the cards
        analysis_result = self.card_analyzer.analyze_meta_cards(meta_cards_df)
        
        return analysis_result
    
    def _calculate_meta_statistics(self, meta_speed: Dict[str, float], 
                                 card_frequencies: Counter,
                                 total_decks: int) -> Dict[str, Any]:
        """Calculate meta statistics"""
        # Calculate most played cards
        most_played = [
            {'card': card, 'count': count}
            for card, count in card_frequencies.most_common(15)
            if card not in self.basic_lands
        ]
        
        # Identify key cards (cards that appear in multiple decks)
        key_cards = [
            card for card, count in card_frequencies.items()
            if count >= total_decks * 0.2  # Present in at least 20% of decks
            and card not in self.basic_lands
        ]
        
        return {
            'total_decks': total_decks,
            'meta_speed': meta_speed,
            'most_played_cards': most_played,
            'key_cards': key_cards
        }

def load_and_preprocess_cards(csv_path: str) -> pd.DataFrame:
    """Load and preprocess card data"""
    try:
        # Define dtypes for better data loading
        dtypes = {
            'name': 'string',
            'full_name': 'string',
            'layout': 'string',
            'type_line': 'string',
            'oracle_text': 'string',
            'power': 'string',  # Keep as string due to */1+* values
            'toughness': 'string',
            'rarity': 'string',
            'set': 'string',
            'collector_number': 'string'
        }
        
        # Load CSV with specified dtypes
        df = pd.read_csv(csv_path, dtype=dtypes)
        
        # Process lists stored as strings
        list_columns = ['colors', 'color_identity', 'keywords', 'produced_mana']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(safe_eval_list)
        
        # Process boolean columns
        bool_columns = [
            'is_creature', 'is_land', 'is_instant_sorcery',
            'is_multicolored', 'has_etb_effect', 'is_legendary'
        ]
        for col in bool_columns:
            if col in df.columns:
                # Convert string representations to actual boolean values
                df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})
        
        # Convert numeric columns
        numeric_columns = ['cmc', 'color_count']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        logger.info(f"Successfully loaded {len(df)} cards")
        return df
        
    except Exception as e:
        logger.error(f"Error loading and preprocessing cards: {e}")
        raise

def load_decklists(directory: str) -> Dict[str, List[str]]:
    """Load decklists from a directory"""
    decklists = {}
    
    try:
        deck_files = [
            f for f in os.listdir(directory) 
            if f.endswith('.txt') and not f.startswith('.')
        ]
        
        for filename in deck_files:
            filepath = os.path.join(directory, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    deck_name = os.path.splitext(filename)[0]
                    mainboard = []
                    sideboard_found = False
                    
                    for line in lines:
                        line = line.strip()
                        
                        # Skip empty lines and comments
                        if not line or line.startswith('#'):
                            continue
                            
                        # Check for sideboard marker
                        if line.lower() == 'sideboard':
                            sideboard_found = True
                            continue
                            
                        if sideboard_found:
                            continue
                            
                        # Parse card entry
                        try:
                            # Handle various formats
                            match = re.match(
                                r'^(?:(\d+)[x]?\s+)?(.+?)(?:\s+[x]?(\d+))?$',
                                line, 
                                re.IGNORECASE
                            )
                            if match:
                                count = int(match.group(1) or match.group(3) or '1')
                                card_name = match.group(2).strip()
                                mainboard.extend([card_name] * count)
                        except Exception as e:
                            logger.warning(f"Could not parse line in {filename}: {line}. Error: {e}")
                    
                    if mainboard:
                        decklists[deck_name] = mainboard
                        
            except Exception as e:
                logger.error(f"Error processing deck file {filename}: {e}")
                continue
        
        logger.info(f"Loaded {len(decklists)} decklists from {len(deck_files)} files")
        return decklists
        
    except Exception as e:
        logger.error(f"Error loading decklists: {e}")
        return {}

def print_meta_analysis_report(meta_analysis: Dict[str, Any]):
    """Print meta analysis report"""
    print("\n=== Magic Format Meta Analysis Report ===\n")
    
    # Meta Speed
    speed_info = meta_analysis['meta_speed']
    print("1. Meta Speed:")
    print(f"   Speed: {speed_info['speed'].title()}")
    print(f"   Average CMC: {speed_info['avg_cmc']:.2f}")
    print(f"   Early Game Ratio: {speed_info['early_game_ratio']*100:.1f}%")
    
    # Card Types
    print("\n2. Card Types in Meta:")
    card_types = meta_analysis['format_characteristics']['card_types']
    for card_type, count in sorted(card_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   {card_type.replace('_', ' ').title()}: {count}")
    
    # Keywords
    print("\n3. Keywords in Meta:")
    keywords = meta_analysis['format_characteristics']['keywords']
    for keyword, count in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"   {keyword.title()}: {count}")
    
    # Subtypes
    print("\n4. Most Common Subtypes:")
    subtypes = meta_analysis['format_characteristics']['subtypes']
    for subtype, count in sorted(subtypes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {subtype.title()}: {count}")
    
    # References
    print("\n5. Type References in Oracle Text:")
    references = meta_analysis['format_characteristics']['references']
    for ref, count in sorted(references.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {ref.replace('_reference', '').replace('_', ' ').title()}: {count}")
    
    # Most Played Cards
    print("\n6. Most Played Cards:")
    for card_info in meta_analysis['meta_statistics']['most_played_cards']:
        print(f"   {card_info['card']}: {card_info['count']} copies")
    
    # Key Cards
    print("\n7. Key Cards (found in multiple decks):")
    key_cards = meta_analysis['meta_statistics']['key_cards']
    for i, card in enumerate(key_cards[:20]):
        print(f"   {card}")
        if i > 0 and i % 10 == 0:
            print("   ...")
            break

def main():
    """Main execution function"""
    try:
        # Parse command-line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Magic Format Meta Analyzer')
        parser.add_argument(
            '--cards', 
            default='data/standard_cards.csv',
            help='Path to cards CSV file'
        )
        parser.add_argument(
            '--decks',
            default='current_standard_decks',
            help='Directory containing decklists'
        )
        parser.add_argument(
            '--output',
            default='meta_analysis_results.json',
            help='Output file for detailed results'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose logging'
        )
        args = parser.parse_args()
        
        # Set up verbose logging if requested
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        # Validate input paths
        if not os.path.exists(args.cards):
            logger.error(f"Cards CSV file not found: {args.cards}")
            return
            
        if not os.path.exists(args.decks):
            logger.error(f"Decks directory not found: {args.decks}")
            return
        
        # Load and process data
        logger.info("Loading card database...")
        card_db = load_and_preprocess_cards(args.cards)
        
        logger.info("Loading decklists...")
        decklists = load_decklists(args.decks)
        
        if not decklists:
            logger.error("No valid decklists found")
            return
        
        # Perform analysis
        logger.info("Analyzing meta...")
        analyzer = MetaAnalyzer(card_db)
        meta_analysis = analyzer.analyze_meta(decklists)
        
        # Print report
        print_meta_analysis_report(meta_analysis)
        
        # Save detailed results
        with open(args.output, 'w') as f:
            # Use a custom JSON serializer to handle non-serializable objects
            def json_serializer(obj):
                if isinstance(obj, pd.DataFrame):
                    return "DataFrame object (not serializable)"
                return str(obj)
                
            json.dump(meta_analysis, f, indent=2, default=json_serializer)
        logger.info(f"Detailed analysis saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error in meta analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()