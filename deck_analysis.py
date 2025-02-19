import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from collections import Counter
import logging
import os
import sys

# Deck Archetype Enum
class DeckArchetype(Enum):
    AGGRO = "aggro"
    MIDRANGE = "midrange"
    CONTROL = "control"
    TEMPO = "tempo"
    COMBO = "combo"
    UNKNOWN = "unknown"

# Utility Functions
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

# Deck Validator Class
class DeckValidator:
    """
    A class to validate Magic: The Gathering decks and provide deck construction insights
    """
    
    STANDARD_DECK_SIZE = 60
    MAX_COPIES = 4
    RECOMMENDED_LAND_RATIO = (0.33, 0.42)  # Min and max recommended land percentages
    
    def __init__(self, card_database: pd.DataFrame):
        """
        Initialize with a card database (from ScryfallFetcher)
        """
        self.card_db = card_database
        self.logger = logging.getLogger(__name__)
    
    def validate_deck(self, decklist: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate a deck against Standard format rules
        Returns: (is_valid, list_of_issues)
        """
        issues = []
        
        # Convert decklist to card counts
        try:
            deck_counts = Counter(decklist)
        except Exception as e:
            issues.append(f"Error processing decklist: {str(e)}")
            return False, issues
            
        # Check deck size
        if len(decklist) < self.STANDARD_DECK_SIZE:
            issues.append(f"Deck contains {len(decklist)} cards. Minimum is {self.STANDARD_DECK_SIZE}")
            
        # Validate card counts and legality
        for card_name, count in deck_counts.items():
            # Check if card exists in database
            card_data = self.card_db[self.card_db['name'] == card_name]
            if len(card_data) == 0:
                issues.append(f"Card not found in database: {card_name}")
                continue
                
            # Check copy limit
            if count > self.MAX_COPIES and 'Basic' not in card_data['type_line'].iloc[0]:
                issues.append(f"Too many copies of {card_name}: {count} (max {self.MAX_COPIES})")
        
        return len(issues) == 0, issues

# Deck Curve Analyzer Class
class DeckCurveAnalyzer:
    """
    Analyzer for deck curve characteristics based on archetype patterns
    """
    
    # Ideal curve characteristics by archetype
    ARCHETYPE_PATTERNS = {
        DeckArchetype.AGGRO: {
            'peak_cmc': 2,
            'max_curve_point': 3,
            'early_game_weight': 0.7,  # 70% of spells should be 1-2 CMC
            'removal_ratio': 0.15,     # 15% removal spells
            'creature_ratio': 0.6      # 60% creatures
        },
        DeckArchetype.MIDRANGE: {
            'peak_cmc': 3,
            'max_curve_point': 5,
            'curve_distribution': 'normal',
            'removal_ratio': 0.25,
            'creature_ratio': 0.5
        },
        DeckArchetype.CONTROL: {
            'peak_cmc': 4,
            'max_curve_point': 7,
            'early_interaction_ratio': 0.3,  # 30% early interaction spells
            'removal_ratio': 0.35,
            'creature_ratio': 0.2
        },
        DeckArchetype.TEMPO: {
            'peak_cmc': 2,
            'max_curve_point': 4,
            'threat_ratio': 0.4,       # 40% threat spells
            'interaction_ratio': 0.3,   # 30% interaction spells
            'creature_ratio': 0.4
        }
    }
    
    def __init__(self):
        self.interaction_keywords = [
            'counter', 'return target', 'destroy target', 'exile target',
            'can\'t attack', 'can\'t block', 'tap target', 'counter target'
        ]
        self.removal_keywords = [
            'destroy', 'exile', 'damage to target', 'dies', '-X/-X'
        ]
        self.cantrip_keywords = [
            'draw a card', 'look at the top', 'scry', 'surveil'
        ]
    
    def analyze_curve(self, deck_cards: pd.DataFrame, decklist: List[str]) -> Dict[str, Any]:
        """
        Analyze mana curve and detect likely archetype
        """
        # Calculate basic curve
        curve = self._calculate_detailed_curve(deck_cards, decklist)
        
        # Analyze card categories
        categories = self._categorize_cards(deck_cards, decklist)
        
        # Detect archetype
        archetype = self._detect_archetype(curve, categories)
        
        # Generate archetype-specific analysis
        analysis = self._analyze_for_archetype(curve, categories, archetype)
        
        return {
            'curve': curve,
            'categories': categories,
            'archetype': archetype,
            'analysis': analysis
        }
    
    def _calculate_detailed_curve(self, deck_cards: pd.DataFrame, decklist: List[str]) -> Dict[str, Any]:
        """
        Calculate detailed mana curve statistics
        """
        curve_counts = Counter()
        spell_counts = Counter()
        
        for _, card in deck_cards.iterrows():
            count = decklist.count(card['name'])
            cmc = int(card['cmc'])
            
            if not card['is_land']:
                curve_counts[cmc] += count
                if self._is_spell(card):
                    spell_counts[cmc] += count
        
        total_spells = sum(curve_counts.values())
        if total_spells == 0:
            return {
                'curve': {},
                'spell_curve': {},
                'percentages': {},
                'avg_cmc': 0,
                'peak_cmc': 0
            }
        
        return {
            'curve': dict(sorted(curve_counts.items())),
            'spell_curve': dict(sorted(spell_counts.items())),
            'percentages': {
                cmc: count/total_spells 
                for cmc, count in curve_counts.items()
            },
            'avg_cmc': sum(cmc * count for cmc, count in curve_counts.items()) / total_spells,
            'peak_cmc': max(curve_counts.items(), key=lambda x: x[1])[0] if curve_counts else 0
        }
    
    def _categorize_cards(self, deck_cards: pd.DataFrame, decklist: List[str]) -> Dict[str, float]:
        """
        Categorize cards by their role in the deck
        """
        categories = {
            'creatures': 0,
            'removal': 0,
            'interaction': 0,
            'cantrips': 0,
            'threats': 0
        }
        
        total_nonland_cards = 0
        
        for _, card in deck_cards.iterrows():
            if card['is_land']:
                continue
                
            count = decklist.count(card['name'])
            total_nonland_cards += count
            
            # Categorize card
            if card['is_creature']:
                categories['creatures'] += count
                if self._is_threat(card):
                    categories['threats'] += count
                    
            if self._has_removal(card):
                categories['removal'] += count
                
            if self._is_interaction(card):
                categories['interaction'] += count
                
            if self._is_cantrip(card):
                categories['cantrips'] += count
        
        # Convert to ratios
        if total_nonland_cards == 0:
            return {category: 0.0 for category in categories}
            
        return {
            category: count/total_nonland_cards 
            for category, count in categories.items()
        }
    
    def _detect_archetype(self, curve: Dict, categories: Dict) -> DeckArchetype:
        """
        Detect the most likely archetype based on curve and card categories
        """
        scores = {archetype: 0 for archetype in DeckArchetype}
        
        # Analyze curve shape
        if curve['peak_cmc'] <= 2 and categories['creatures'] >= 0.5:
            scores[DeckArchetype.AGGRO] += 2
            
        if 2 <= curve['peak_cmc'] <= 3 and 0.4 <= categories['creatures'] <= 0.6:
            scores[DeckArchetype.MIDRANGE] += 2
            
        if curve['peak_cmc'] >= 4 and categories['interaction'] >= 0.25:
            scores[DeckArchetype.CONTROL] += 2
            
        if curve['peak_cmc'] <= 3 and categories['interaction'] >= 0.25:
            scores[DeckArchetype.TEMPO] += 2
        
        # Analyze card ratios
        if categories['removal'] >= 0.3:
            scores[DeckArchetype.CONTROL] += 1
            
        if categories['cantrips'] >= 0.15:
            scores[DeckArchetype.CONTROL] += 1
            scores[DeckArchetype.TEMPO] += 1
            
        if categories['threats'] >= 0.4:
            scores[DeckArchetype.AGGRO] += 1
            
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _analyze_for_archetype(self, curve: Dict, categories: Dict, 
                             archetype: DeckArchetype) -> Dict[str, Any]:
        """
        Generate archetype-specific analysis and recommendations
        """
        pattern = self.ARCHETYPE_PATTERNS.get(archetype)
        if not pattern:
            return {}
            
        recommendations = []
        
        # Check curve against archetype pattern
        if curve['peak_cmc'] != pattern['peak_cmc']:
            recommendations.append(
                f"Consider adjusting curve peak to {pattern['peak_cmc']} CMC "
                f"for optimal {archetype.value} performance"
            )
        
        # Check ratios
        for category, target_ratio in pattern.items():
            if category.endswith('_ratio'):
                actual_ratio = categories.get(category.replace('_ratio', ''), 0)
                if abs(actual_ratio - target_ratio) > 0.1:
                    recommendations.append(
                        f"Adjust {category.replace('_ratio', '')} count to around "
                        f"{target_ratio:.0%} for {archetype.value}"
                    )
        
        return {
            'archetype_fit': self._calculate_archetype_fit(curve, categories, pattern),
            'recommendations': recommendations,
            'ideal_pattern': pattern
        }
    
    def _calculate_archetype_fit(self, curve: Dict, categories: Dict, 
                               pattern: Dict) -> float:
        """
        Calculate how well the deck fits its detected archetype
        """
        differences = []
        
        for category, target in pattern.items():
            if category.endswith('_ratio'):
                actual = categories.get(category.replace('_ratio', ''), 0)
                differences.append(abs(actual - target))
                
        if differences:
            return 1 - (sum(differences) / len(differences))
        return 0.5
    
    def _is_spell(self, card: pd.Series) -> bool:
        return not card['is_land']
    
    def _is_threat(self, card: pd.Series) -> bool:
        return (card['is_creature'] and 
                (card['power'] is not None and float(card['power']) >= 3))
    
    def _has_removal(self, card: pd.Series) -> bool:
        return any(keyword in (card['oracle_text'] or '').lower() 
                  for keyword in self.removal_keywords)
    
    def _is_interaction(self, card: pd.Series) -> bool:
        return any(keyword in (card['oracle_text'] or '').lower() 
                  for keyword in self.interaction_keywords)
    
    def _is_cantrip(self, card: pd.Series) -> bool:
        return any(keyword in (card['oracle_text'] or '').lower() 
                  for keyword in self.cantrip_keywords)

# Main Analysis Function
def analyze_deck(cards_df: pd.DataFrame, decklist_path: str):
    """
    Analyze a deck from a given decklist file
    """
    # Load decklist
    mainboard, _ = load_decklist(decklist_path)
    
    # Initialize analyzers
    validator = DeckValidator(cards_df)
    curve_analyzer = DeckCurveAnalyzer()
    
    # Validate deck
    is_valid, issues = validator.validate_deck(mainboard)
    print(f"=== Deck Analysis: {os.path.basename(decklist_path)} ===\n")
    print("Deck Validation:")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("✓ Mainboard is valid")
    
    # Analyze curve and archetype
    deck_cards = cards_df[cards_df['name'].isin(mainboard)]
    analysis = curve_analyzer.analyze_curve(deck_cards, mainboard)
    
    print(f"\nDetected Archetype: {analysis['archetype'].value}")
    print(f"Archetype Fit Score: {analysis['analysis']['archetype_fit']:.2%}")
    
    print("\nMana Curve:")
    for cmc, count in analysis['curve']['curve'].items():
        print(f"CMC {cmc}: {'█' * count} {count} cards")
    
    print("\nCard Categories:")
    for category, ratio in analysis['categories'].items():
        print(f"{category.title()}: {ratio:.1%}")
    
    if analysis['analysis']['recommendations']:
        print("\nArchetype-based Recommendations:")
        for rec in analysis['analysis']['recommendations']:
            print(f"- {rec}")

# Main Execution
def main():
    # Ensure a deck file is provided
    if len(sys.argv) < 2:
        print("Usage: python deck_analysis.py <deck_file_path>")
        sys.exit(1)
    
    # Load card database
    cards_df = pd.read_csv('data/standard_cards.csv')
    
    # Analyze the specified deck
    decklist_path = sys.argv[1]
    analyze_deck(cards_df, decklist_path)

if __name__ == "__main__":
    main()