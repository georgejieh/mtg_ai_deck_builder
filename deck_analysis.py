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
        if card['is_creature']:
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
            
            # Verify cards in database
            deck_cards = self.card_db[self.card_db['name'].isin(decklist)]
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
            logger.error("Error in deck analysis: {}".format(str(e)))
            raise
    
    def _calculate_deck_statistics(self, deck_cards: pd.DataFrame, decklist: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive deck statistics
        """
        nonland_cards = deck_cards[~deck_cards['is_land']]
        
        # Calculate various ratios and statistics
        deck_stats = {
            'land_ratio': len(deck_cards[deck_cards['is_land']]) / len(decklist),
            'creature_ratio': len(nonland_cards[nonland_cards['is_creature']]) / len(nonland_cards),
            'avg_cmc': nonland_cards['cmc'].mean(),
            'median_cmc': nonland_cards['cmc'].median(),
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
        
        # Interaction ratio
        interaction_patterns = [
            r'counter target', r'destroy target', r'exile target', 
            r'tap target', r'can\'t attack', r'can\'t block'
        ]
        interaction_cards = nonland_cards[
            nonland_cards['oracle_text'].apply(
                lambda x: any(re.search(pattern, str(x).lower()) for pattern in interaction_patterns)
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
        entropy = -sum((count/total_colors) * np.log(count/total_colors) 
                       for count in color_counts.values())
        
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
    """
    with open(filepath, 'r') as f:
        contents = f.read().strip().split('\n\n')
    
    # If no double newline, assume all cards are mainboard
    if len(contents) == 1:
        mainboard = contents[0].split('\n')
        sideboard = []
    else:
        mainboard = contents[0].split('\n')
        sideboard = contents[1].split('\n')
    
    # Process card names, removing count
    def process_cards(card_list):
        processed_cards = []
        for card_line in card_list:
            # Split into count and card name, take the card name
            card_name = ' '.join(card_line.split()[1:])
            # Repeat the card name by its count
            count = int(card_line.split()[0])
            processed_cards.extend([card_name] * count)
        return processed_cards
    
    return process_cards(mainboard), process_cards(sideboard)

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
        bar = '█' * count
        print("{} CMC: {} {} cards".format(cmc, bar, count))
    
    # Deck Composition Analysis
    print("\nDeck Composition:")
    print("Interaction Density: {:.0f}%".format(analysis['statistics']['interaction_ratio'] * 100))
    print("Removal Density: {:.0f}%".format(analysis['statistics']['removal_ratio'] * 100))
    print("Card Advantage Potential: {:.0f}%".format(analysis['statistics']['card_advantage_ratio'] * 100))
    
    # Mechanics Breakdown
    print("\nKey Deck Mechanics:")
    sorted_mechanics = sorted(analysis['mechanics'].items(), key=lambda x: x[1], reverse=True)
    top_mechanics = sorted_mechanics[:5]  # Top 5 mechanics
    
    mechanic_descriptions = {
        'card_draw': "Ability to draw additional cards",
        'card_selection': "Ability to look at and potentially rearrange top cards",
        'removal': "Spells that eliminate opponent's creatures or threats",
        'surveil': "Look at top cards and put them in graveyard or back on top",
        'token_generation': "Create additional creature tokens",
        'big_creature': "Powerful, high-impact creatures",
        'threat': "Creatures that demand an immediate answer",
        'life_gain': "Ability to restore life points"
    }
    
    for mechanic, count in top_mechanics:
        desc = mechanic_descriptions.get(mechanic, "Unique mechanical interaction")
        print("- {}: {} ({})".format(mechanic.replace('_', ' ').title(), count, desc))
    
    # Missing Cards Warning
    missing_cards = set(mainboard) - set(analysis['verified_cards'])
    if missing_cards:
        print("\nWarning - Cards not found in database:")
        for card in missing_cards:
            print("- {}".format(card))

def main():
    # Ensure a deck file is provided
    if len(sys.argv) < 2:
        print("Usage: python deck_analysis.py <deck_file_path>")
        sys.exit(1)
    
    # Load card database
    try:
        cards_df = pd.read_csv('data/standard_cards.csv')
    except FileNotFoundError:
        print("Error: Card database file not found. Please ensure 'data/standard_cards.csv' exists.")
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