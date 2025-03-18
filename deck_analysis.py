import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from collections import Counter, defaultdict
import logging
import os
import sys
import re

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeckArchetype(Enum):
    AGGRO = "aggro"
    MIDRANGE = "midrange"
    CONTROL = "control"
    TEMPO = "tempo"
    COMBO = "combo"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"

    @classmethod
    def get_characteristics(cls) -> Dict['DeckArchetype', Dict[str, Tuple[float, float]]]:
        """
        Provides dynamic characteristics for each archetype
        These are more flexible than hardcoded values
        """
        return {
            cls.AGGRO: {
                'creature_ratio': (0.45, 0.75),
                'removal_ratio': (0.1, 0.3),
                'curve_peak': (1, 2),
                'avg_cmc': (1.5, 2.5),
                'interaction_ratio': (0.0, 0.2),
                'card_advantage_ratio': (0.0, 0.15)
            },
            cls.MIDRANGE: {
                'creature_ratio': (0.35, 0.55),
                'removal_ratio': (0.15, 0.35),
                'curve_peak': (2, 4),
                'avg_cmc': (2.5, 3.5),
                'interaction_ratio': (0.1, 0.3),
                'card_advantage_ratio': (0.1, 0.3)
            },
            cls.CONTROL: {
                'creature_ratio': (0.1, 0.3),
                'removal_ratio': (0.25, 0.45),
                'curve_peak': (2, 5),
                'avg_cmc': (3.0, 4.5),
                'interaction_ratio': (0.25, 0.5),
                'card_advantage_ratio': (0.2, 0.4)
            },
            cls.TEMPO: {
                'creature_ratio': (0.3, 0.5),
                'removal_ratio': (0.15, 0.35),
                'curve_peak': (1, 3),
                'avg_cmc': (2.0, 3.0),
                'interaction_ratio': (0.2, 0.4),
                'card_advantage_ratio': (0.1, 0.25)
            }
        }

def safe_eval_list(val):
    """Safely convert string representation of list to actual list"""
    if pd.isna(val) or val is None:
        return []
        
    if isinstance(val, list):
        return val
        
    if isinstance(val, str):
        try:
            if val.startswith('[') and val.endswith(']'):
                # Parse string representation of list manually to avoid unsafe eval
                # Remove brackets and split by commas
                inner = val[1:-1].strip()
                if not inner:
                    return []
                    
                items = []
                for item in inner.split(','):
                    clean_item = item.strip().strip('\'"')
                    if clean_item:
                        items.append(clean_item)
                return items
            else:
                # Not a list representation, return as single item list
                return [val]
        except Exception as e:
            logger.warning(f"Error parsing list from string: {val}. Error: {e}")
            return []
    
    # For any other type, return empty list
    return []

class AdvancedDeckAnalyzer:
    """
    A comprehensive deck analysis system with dynamic archetype detection
    """
    def __init__(self, card_database: pd.DataFrame):
        self.card_db = card_database
        self.archetype_characteristics = DeckArchetype.get_characteristics()
        self.decklist = []
        
    def _extract_card_mechanics(self, card: pd.Series) -> Dict[str, int]:
        """
        Extract detailed mechanics from a card
        """
        mechanics = defaultdict(int)
        oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
        
        # Comprehensive mechanic patterns
        mechanic_patterns = {
            'card_draw': [r'draw \d+ cards?', r'look at the top \d+ cards?'],
            'removal': [r'destroy target', r'exile target', r'deals? \d+ damage to', r'-\d+/-\d+'],
            'counter_spell': [r'counter target spell', r'counter target'],
            'ramp': [r'add \d+ mana', r'search your library for a land'],
            'token_generation': [r'create.*token', r'create a.*token'],
            'life_gain': [r'gain \d+ life'],
            'scry': [r'scry \d+'],
            'surveil': [r'surveil \d+'],
            'card_selection': [r'look at', r'reveal', r'choose'],
            'recursion': [r'return.*from your graveyard'],
        }
        
        # Check card type mechanics
        if card['is_creature'] == True:
            mechanics['creature'] += 1
            
            # Power-based mechanics
            try:
                power = float(card['power']) if pd.notna(card['power']) else 0
                toughness = float(card['toughness']) if pd.notna(card['toughness']) else 0
                
                if power >= 3:
                    mechanics['threat'] += 1
                if power + toughness >= 5:
                    mechanics['big_creature'] += 1
            except (ValueError, TypeError):
                pass
        
        # Special handling for Room cards
        if 'Room' in str(card['type_line']):
            mechanics['room'] += 1
            
            if 'unlock' in oracle_text:
                mechanics['unlock_mechanic'] += 1
                
            if 'door' in oracle_text:
                mechanics['door_mechanic'] += 1
        
        # Check keyword mechanics
        if isinstance(card['keywords'], list):
            for keyword in card['keywords']:
                mechanics[keyword.lower()] += 1
        
        # Check text-based mechanics
        for mechanic, patterns in mechanic_patterns.items():
            if any(re.search(pattern, oracle_text) for pattern in patterns):
                mechanics[mechanic] += 1
        
        return dict(mechanics)
        
    def analyze_deck(self, decklist: List[str]) -> Dict[str, Any]:
        """
        Comprehensive deck analysis with dynamic archetype detection
        """
        try:
            # Store the original decklist for later use in mana curve calculation
            self.decklist = decklist
            
            # Verify cards in database - enhanced to handle split cards
            deck_cards = self._match_cards_in_database(decklist)
            verified_cards = deck_cards['name'].tolist()
            
            # Basic deck statistics
            stats = self._calculate_deck_statistics(deck_cards, decklist)
            
            # Card mechanics analysis
            mechanics_breakdown = self._analyze_deck_mechanics(deck_cards)
            
            # Archetype detection
            archetype_scores = self._detect_archetype(stats)
            
            # Final analysis
            analysis = {
                'deck_size': len(decklist),
                'statistics': stats,
                'mechanics': mechanics_breakdown,
                'archetype_scores': archetype_scores,
                'primary_archetype': max(archetype_scores.items(), key=lambda x: x[1])[0],
                'verified_cards': verified_cards
            }
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error in deck analysis: {str(e)}")
            raise
    
    def _match_cards_in_database(self, decklist: List[str]) -> pd.DataFrame:
        """
        Enhanced card matching to handle split cards and variations
        with improved handling of different separator formats
        """
        # Create a set of unique cards for faster lookup
        unique_cards = set(decklist)
        matched_cards = pd.DataFrame()
        unmatched_cards = set(unique_cards)
        
        # First try exact name match
        exact_matches = self.card_db[self.card_db['name'].isin(unique_cards)]
        matched_cards = pd.concat([matched_cards, exact_matches])
        
        # Track which cards were matched
        matched_names = set()
        for name in matched_cards['name']:
            # Ensure we're adding a hashable type (string) to the set
            if isinstance(name, str):
                matched_names.add(name)
            elif isinstance(name, list):
                # If for some reason a name is a list, convert it to a string
                matched_names.add(str(name))
        
        # Find remaining unmatched cards
        unmatched_cards = {card for card in unique_cards if card not in matched_names}
        
        # For remaining cards, try matching against the full_name field
        if unmatched_cards:
            # Look for cards that might be split cards with different separators
            for card_name in list(unmatched_cards):  # Use list to allow modification during iteration
                # Try variations of split card separators
                if isinstance(card_name, str) and '/' in card_name:
                    # Convert "Name1/Name2" to possible database formats
                    parts = card_name.split('/')
                    if len(parts) == 2:
                        name1, name2 = [p.strip() for p in parts]
                        
                        # Try different separator formats
                        alt_formats = [
                            f"{name1} // {name2}",  # Standard format with double slash
                            f"{name1}/{name2}",     # Single slash no spaces
                            f"{name1} / {name2}"    # Single slash with spaces
                        ]
                        
                        found = False
                        # Check for matches in full_name
                        for alt_format in alt_formats:
                            format_matches = self.card_db[self.card_db['full_name'] == alt_format]
                            if not format_matches.empty:
                                matched_cards = pd.concat([matched_cards, format_matches])
                                if card_name in unmatched_cards:
                                    unmatched_cards.remove(card_name)
                                found = True
                                break
                                
                        if found:
                            continue
                
                # Also try matching against just the front face
                if isinstance(card_name, str):
                    front_face_matches = self.card_db[
                        self.card_db['full_name'].str.split(' // ').str[0] == card_name
                    ]
                    
                    if not front_face_matches.empty:
                        matched_cards = pd.concat([matched_cards, front_face_matches])
                        if card_name in unmatched_cards:
                            unmatched_cards.remove(card_name)
        
        # Remove duplicates that might have been added
        return matched_cards.drop_duplicates()
    
    def _calculate_deck_statistics(self, deck_cards: pd.DataFrame, decklist: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive deck statistics
        """
        # Improved boolean filtering
        nonland_cards = deck_cards[deck_cards['is_land'] != True]
        
        # Calculate various ratios and statistics
        deck_stats = {
            'land_ratio': len(deck_cards[deck_cards['is_land'] == True]) / len(decklist) if len(decklist) > 0 else 0,
            'creature_ratio': len(nonland_cards[nonland_cards['is_creature'] == True]) / len(nonland_cards) if len(nonland_cards) > 0 else 0,
            'avg_cmc': nonland_cards['cmc'].mean() if len(nonland_cards) > 0 else 0,
            'median_cmc': nonland_cards['cmc'].median() if len(nonland_cards) > 0 else 0,
            'color_diversity': self._calculate_color_diversity(deck_cards),
            'curve': self._calculate_mana_curve(nonland_cards)
        }
        
        # Calculate advanced ratios
        deck_stats.update(self._calculate_advanced_ratios(nonland_cards, decklist))
        
        return deck_stats
    
    def _calculate_mana_curve(self, nonland_cards: pd.DataFrame) -> Dict[int, int]:
        """
        Calculate detailed mana curve considering card copies
        """
        # Create a dictionary to track card counts
        card_counts = {}
        for card_name in nonland_cards['name']:
            # Count the number of copies in the original decklist
            count = self.decklist.count(card_name)
            
            # Also check if this is a split card front face
            if count == 0:
                # This might be a split card, check if the name appears in any full_name
                full_names = nonland_cards[nonland_cards['name'] == card_name]['full_name']
                if not full_names.empty:
                    for full_name in full_names:
                        if isinstance(full_name, str) and ' // ' in full_name:
                            front_face = full_name.split(' // ')[0]
                            count = self.decklist.count(front_face)
                            
                            # Also check for alternate separator format
                            if count == 0 and '/' in front_face:
                                # This is unlikely but let's be thorough
                                name1, name2 = front_face.split('/')[0], full_name.split(' // ')[1]
                                alt_format = f"{name1}/{name2}"
                                count = self.decklist.count(alt_format)
            
            # If we still have no count, try checking for variations with slash
            if count == 0:
                # Try to find corresponding split card in decklist
                for deck_card in self.decklist:
                    if isinstance(deck_card, str) and '/' in deck_card:
                        front_face = deck_card.split('/')[0].strip()
                        if front_face == card_name:
                            count = self.decklist.count(deck_card)
                            break
            
            # If we still have no count, skip this card
            if count == 0:
                continue
            
            # Get the CMC for this card
            card_cmc = nonland_cards[nonland_cards['name'] == card_name]['cmc'].iloc[0]
            
            # Round CMC to nearest integer
            cmc_int = int(round(card_cmc))
            
            # Add to curve, respecting the number of copies
            card_counts[cmc_int] = card_counts.get(cmc_int, 0) + count
        
        return card_counts
        
    def _calculate_advanced_ratios(self, nonland_cards: pd.DataFrame, decklist: List[str]) -> Dict[str, float]:
        """
        Calculate advanced deck ratios
        """
        advanced_ratios = {}
        
        # Prevent division by zero
        if len(nonland_cards) == 0:
            return {
                'interaction_ratio': 0,
                'removal_ratio': 0,
                'card_advantage_ratio': 0
            }
        
        # Interaction ratio
        interaction_patterns = [
            r'counter target', r'destroy target', r'exile target', 
            r'tap target', r'can\'t attack', r'can\'t block'
        ]
        interaction_cards = nonland_cards[
            nonland_cards['oracle_text'].apply(
                lambda x: any(re.search(pattern, str(x).lower()) for pattern in interaction_patterns)
                if pd.notna(x) else False
            )
        ]
        advanced_ratios['interaction_ratio'] = len(interaction_cards) / len(nonland_cards)
        
        # Removal ratio
        removal_patterns = [
            r'destroy target', r'exile target', r'deals? \d+ damage to', 
            r'target creature gets -\d+/-\d+'
        ]
        removal_cards = nonland_cards[
            nonland_cards['oracle_text'].apply(
                lambda x: any(re.search(pattern, str(x).lower()) for pattern in removal_patterns)
                if pd.notna(x) else False
            )
        ]
        advanced_ratios['removal_ratio'] = len(removal_cards) / len(nonland_cards)
        
        # Card advantage ratio
        card_advantage_patterns = [
            r'draw \d+ cards?', r'return.*from your graveyard', 
            r'search your library', r'look at the top \d+ cards?'
        ]
        card_advantage_cards = nonland_cards[
            nonland_cards['oracle_text'].apply(
                lambda x: any(re.search(pattern, str(x).lower()) for pattern in card_advantage_patterns)
                if pd.notna(x) else False
            )
        ]
        advanced_ratios['card_advantage_ratio'] = len(card_advantage_cards) / len(nonland_cards)
        
        return advanced_ratios
    
    def _calculate_color_diversity(self, deck_cards: pd.DataFrame) -> float:
        """
        Calculate color diversity of the deck
        """
        all_colors = []
        for _, card in deck_cards.iterrows():
            if isinstance(card['colors'], list):
                all_colors.extend(card['colors'])
        
        # Use entropy to measure color diversity
        if not all_colors:
            return 0
        
        color_counts = Counter(all_colors)
        total_colors = len(all_colors)
        
        # Prevent log(0) issues
        entropy = -sum((count/total_colors) * np.log(count/total_colors) 
                      for count in color_counts.values() if count > 0)
        
        return entropy
    
    def _analyze_deck_mechanics(self, deck_cards: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze mechanics across the entire deck
        """
        mechanics_breakdown = defaultdict(int)
        
        for _, card in deck_cards.iterrows():
            card_mechanics = self._extract_card_mechanics(card)
            for mechanic, count in card_mechanics.items():
                mechanics_breakdown[mechanic] += count
        
        return dict(mechanics_breakdown)
    
    def _detect_archetype(self, stats: Dict[str, float]) -> Dict[DeckArchetype, float]:
        """
        Detect deck archetype with confidence scores
        """
        archetype_scores = {}
        
        for archetype, characteristics in self.archetype_characteristics.items():
            score = 0
            total_checks = 0
            
            # Check each characteristic
            for characteristic, (min_val, max_val) in characteristics.items():
                if characteristic in stats:
                    value = stats.get(characteristic, 0)
                    if min_val <= value <= max_val:
                        score += 1
                    total_checks += 1
            
            # Calculate archetype confidence
            archetype_scores[archetype] = score / total_checks if total_checks > 0 else 0
        
        # Check for potential hybrid archetype
        sorted_scores = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1][1] > 0.7 * sorted_scores[0][1]:
            archetype_scores[DeckArchetype.HYBRID] = (sorted_scores[0][1] + sorted_scores[1][1]) / 2
        
        return archetype_scores

def load_decklist(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Load decklist from a file, separating mainboard and sideboard
    Enhanced to better handle different file formats and split cards
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    mainboard = []
    sideboard = []
    in_sideboard = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
            
        # Check for sideboard marker
        if line.lower() == 'sideboard':
            in_sideboard = True
            continue
        
        # Parse card entry
        try:
            # Handle various formats like "4 Card Name" or "Card Name x4"
            match = re.match(r'^(?:(\d+)[x]?\s+)?(.+?)(?:\s+[x]?(\d+))?$', line, re.IGNORECASE)
            if match:
                count = int(match.group(1) or match.group(3) or '1')
                card_name = match.group(2).strip()
                
                # Add card to appropriate list
                if in_sideboard:
                    sideboard.extend([card_name] * count)
                else:
                    mainboard.extend([card_name] * count)
        except Exception as e:
            logger.warning(f"Could not parse line: {line}. Error: {e}")
    
    return mainboard, sideboard

def analyze_deck(cards_df: pd.DataFrame, decklist_path: str):
    """
    Advanced deck analysis function
    """
    # Validate input
    if not os.path.exists(decklist_path):
        raise FileNotFoundError("Decklist file not found: {}".format(decklist_path))
    
    # Load decklist
    try:
        mainboard, sideboard = load_decklist(decklist_path)
    except Exception as e:
        raise ValueError("Error loading decklist: {}".format(e))
    
    # Validate mainboard
    if not mainboard:
        raise ValueError("Decklist is empty")
    
    # Initialize analyzer
    analyzer = AdvancedDeckAnalyzer(cards_df)
    
    # Analyze mainboard
    try:
        analysis = analyzer.analyze_deck(mainboard)
    except Exception as e:
        raise RuntimeError("Deck analysis failed: {}".format(e))
    
    # Display results with enhanced readability
    print("\n=== Deck Analysis: {} ===\n".format(os.path.basename(decklist_path)))
    
    # Deck Overview
    print("- Total Cards: {}".format(analysis['deck_size']))
    print("- Lands: {:.0f}%".format(analysis['statistics']['land_ratio'] * 100))
    print("- Creatures: {:.0f}%".format(analysis['statistics']['creature_ratio'] * 100))
    
    # Room cards detection
    if 'room' in analysis['mechanics']:
        room_count = analysis['mechanics']['room']
        print("- Room Cards: {}".format(room_count))
    
    # Archetype Analysis
    print("\nArchetype Analysis:")
    archetype_descriptions = {
        DeckArchetype.AGGRO: "Aggressive strategy focused on quick, low-cost threats",
        DeckArchetype.MIDRANGE: "Balanced strategy with mid-cost creatures and flexible answers",
        DeckArchetype.CONTROL: "Defensive strategy focused on controlling the game and answering threats",
        DeckArchetype.TEMPO: "Strategy that balances threats with interaction to maintain board advantage",
        DeckArchetype.COMBO: "Strategy relying on specific card combinations to win",
        DeckArchetype.HYBRID: "Mixed strategy combining elements of multiple archetypes"
    }
    
    sorted_archetypes = sorted(analysis['archetype_scores'].items(), key=lambda x: x[1], reverse=True)
    top_archetypes = [arch for arch, score in sorted_archetypes if score > 0]
    
    print("Primary Strategy: {}".format(analysis['primary_archetype'].value.title()))
    if len(top_archetypes) > 1:
        print("Secondary Strategies: {}".format(
            ", ".join(arch.value.title() for arch in top_archetypes[1:] if arch != analysis['primary_archetype'])
        ))
    
    # Detailed Strategy Description
    primary_archetype = analysis['primary_archetype']
    print("- {}".format(archetype_descriptions.get(primary_archetype, "Unique strategy")))
    
    # Mana Curve Analysis
    print("\nMana Curve Insights:")
    curve = analysis['statistics']['curve']
    print("Average Converted Mana Cost (CMC): {:.2f}".format(analysis['statistics']['avg_cmc']))
    print("Median Converted Mana Cost (CMC): {:.2f}".format(analysis['statistics']['median_cmc']))
    
    print("Spell Distribution by Mana Cost:")
    for cmc, count in sorted(curve.items()):
        bar = '█' * min(count, 30)  # Limit bar length for display purposes
        print("{} CMC: {} {} cards".format(cmc, bar, count))
    
    # Deck Composition Analysis
    print("\nDeck Composition:")
    print("Interaction Density: {:.0f}%".format(analysis['statistics']['interaction_ratio'] * 100))
    print("Removal Density: {:.0f}%".format(analysis['statistics']['removal_ratio'] * 100))
    print("Card Advantage Potential: {:.0f}%".format(analysis['statistics']['card_advantage_ratio'] * 100))
    
    # Mechanics Breakdown
    print("\nKey Deck Mechanics:")
    sorted_mechanics = sorted(analysis['mechanics'].items(), key=lambda x: x[1], reverse=True)
    top_mechanics = sorted_mechanics[:8]  # Top 8 mechanics
    
    mechanic_descriptions = {
        'card_draw': "Ability to draw additional cards",
        'card_selection': "Ability to look at and potentially rearrange top cards",
        'removal': "Spells that eliminate opponent's creatures or threats",
        'surveil': "Look at top cards and put them in graveyard or back on top",
        'token_generation': "Create additional creature tokens",
        'big_creature': "Powerful, high-impact creatures",
        'threat': "Creatures that demand an immediate answer",
        'life_gain': "Ability to restore life points",
        'room': "Room enchantments that provide utility",
        'unlock_mechanic': "Ability to unlock room doors for additional effects",
        'door_mechanic': "Mechanics involving door unlocking and room transitions"
    }
    
    for mechanic, count in top_mechanics:
        desc = mechanic_descriptions.get(mechanic, "Unique mechanical interaction")
        print("- {}: {} ({})".format(mechanic.replace('_', ' ').title(), count, desc))
    
    # Missing Cards Warning
    missing_cards = set(mainboard) - set(analysis['verified_cards'])
    if missing_cards:
        print("\nWarning - Cards not found in database:")
        for card in missing_cards:
            # Check if this might be a split card with wrong separator
            if isinstance(card, str) and '/' in card:
                parts = card.split('/')
                if len(parts) == 2:
                    name1, name2 = [p.strip() for p in parts]
                    alt_format = f"{name1} // {name2}"
                    print(f"- {card} (might be '{alt_format}' in database)")
            else:
                print(f"- {card}")

def main():
    # Ensure a deck file is provided
    if len(sys.argv) < 2:
        print("Usage: python deck_analysis.py <deck_file_path>")
        sys.exit(1)
    
    # Load card database
    try:
        cards_df = pd.read_csv('data/standard_cards.csv')
        
        # Convert boolean columns to actual booleans if they are strings
        bool_columns = ['is_creature', 'is_land', 'is_instant_sorcery', 
                        'is_multicolored', 'has_etb_effect', 'is_legendary']
        for col in bool_columns:
            if col in cards_df.columns:
                cards_df[col] = cards_df[col].map({'True': True, 'False': False, True: True, False: False})
        
        # Convert lists stored as strings to actual lists using safe parsing
        list_columns = ['colors', 'color_identity', 'keywords', 'produced_mana']
        for col in list_columns:
            if col in cards_df.columns:
                cards_df[col] = cards_df[col].apply(safe_eval_list)
        
    except FileNotFoundError:
        print("Error: Card database file not found. Please ensure 'data/standard_cards.csv' exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading card database: {e}")
        sys.exit(1)
    
    # Analyze the specified deck
    decklist_path = sys.argv[1]
    
    try:
        analyze_deck(cards_df, decklist_path)
    except Exception as e:
        print("An error occurred during deck analysis: {}".format(e))
        sys.exit(1)

if __name__ == "__main__":
    main()