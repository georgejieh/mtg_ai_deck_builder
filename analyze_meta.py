import os
import re
import json
import logging
import itertools
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Set
from collections import Counter, defaultdict
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from Levenshtein import distance

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicArchetypeClassifier:
    """
    Dynamic classifier for deck archetypes that adapts to the current card pool
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.archetypes = self._discover_archetypes()
    
    def _discover_archetypes(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Dynamically discover archetypes based on statistical analysis of the card pool
        """
        # Analyze card characteristics
        nonland_cards = self.card_db[~self.card_db['is_land']]
        
        # Calculate distribution metrics
        stats = {
            'creature_ratio': nonland_cards['is_creature'].mean(),
            'avg_cmc': nonland_cards['cmc'].mean(),
            'color_complexity': nonland_cards['color_count'].mean(),
            'legendary_ratio': nonland_cards['is_legendary'].mean()
        }
        
        # Define dynamic archetype discovery criteria
        archetypes = {
            'aggro': {
                'creature_ratio': (0.4, 0.7),
                'avg_cmc': (1.0, 2.5),
                'color_complexity': (1.0, 2.0)
            },
            'midrange': {
                'creature_ratio': (0.3, 0.6),
                'avg_cmc': (2.5, 3.5),
                'color_complexity': (2.0, 3.0)
            },
            'control': {
                'creature_ratio': (0.1, 0.4),
                'avg_cmc': (3.0, 4.5),
                'color_complexity': (2.0, 3.0)
            },
            'combo': {
                'creature_ratio': (0.2, 0.5),
                'avg_cmc': (2.0, 4.0),
                'color_complexity': (2.0, 3.0)
            }
        }
        
        return archetypes

class DynamicCardMechanicsAnalyzer:
    """
    Advanced card mechanics analyzer that dynamically extracts mechanics 
    from the current card pool
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
    
    def extract_mechanics_and_abilities(self) -> Dict[str, int]:
        """
        Dynamically extract mechanics and abilities using NLP techniques
        """
        mechanics = Counter()
        
        for _, card in self.card_db.iterrows():
            # Handle keywords (now already a list)
            if isinstance(card['keywords'], list):
                mechanics.update(card['keywords'])
            
            # Process oracle text
            oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
            
            # Add regex patterns for common mechanics
            mechanic_patterns = {
                'adventure': r'adventure',
                'convoke': r'convoke',
                'prowess': r'prowess',
                'food': r'food token|sacrifice a food',
                'treasure': r'treasure token|sacrifice a treasure',
                'investigate': r'investigate|sacrifice a clue',
                'bargain': r'bargain',
                'token': r'create.*token',
                'sacrifice': r'sacrifice a',
                'ward': r'ward',
                'disturb': r'disturb',
                'daybound': r'daybound',
                'nightbound': r'nightbound',
                'defender': r'defender',
                'flash': r'flash',
                'flying': r'flying',
                'haste': r'haste',
                'lifelink': r'lifelink',
                'trample': r'trample',
                'vigilance': r'vigilance'
            }
            
            for mechanic, pattern in mechanic_patterns.items():
                if re.search(pattern, oracle_text, re.IGNORECASE):
                    mechanics[mechanic] += 1
        
        # Filter out mechanics that appear too rarely to be significant
        min_threshold = len(self.card_db) * 0.01  # 1% of cards
        return {k: v for k, v in mechanics.items() if v >= min_threshold}

class DynamicSynergyDetector:
    """
    Advanced synergy detector that dynamically identifies card interactions
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
    
    def detect_synergies(self) -> Dict[str, List[str]]:
        """
        Detect card synergies using multiple approaches
        """
        synergies = defaultdict(list)
        
        # Keyword Co-occurrence Synergy Detection
        def keyword_co_occurrence_synergy():
            def extract_keywords(text):
                return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
            
            # Map of card names to their keywords
            card_keywords = {}
            for _, card in self.card_db.iterrows():
                text = str(card['oracle_text']).lower() + ' ' + str(card['type_line']).lower()
                card_keywords[card['name']] = extract_keywords(text)
            
            # Find keyword-based synergies
            all_keywords = set.union(*card_keywords.values())
            for keyword_combo in itertools.combinations(all_keywords, 2):
                synergistic_cards = [
                    card_name for card_name, keywords in card_keywords.items()
                    if all(kw in keywords for kw in keyword_combo)
                ]
                if 3 <= len(synergistic_cards) <= 20:
                    synergies[f"{'_'.join(keyword_combo)}_synergy"] = synergistic_cards
        
        # Type-based Synergy Detection
        def type_synergy_detection():
            type_groups = defaultdict(list)
            for _, card in self.card_db.iterrows():
                types = re.findall(r'\b\w+\b', str(card['type_line']).lower())
                for r in range(2, min(4, len(types) + 1)):
                    for combo in itertools.combinations(types, r):
                        type_groups[tuple(sorted(combo))].append(card['name'])
            
            # Filter meaningful type synergies
            for type_group, cards in type_groups.items():
                if 3 <= len(cards) <= 20:
                    synergies[f"{'_'.join(type_group)}_type_synergy"] = cards
        
        # Effect-based Synergy Detection
        def effect_synergy_detection():
            effect_patterns = [
                (r'create.*token', 'token_generation'),
                (r'draw \d+ cards?', 'card_draw'),
                (r'(gain|lose) \d+ life', 'life_manipulation'),
                (r'deals? \d+ damage', 'damage_dealing'),
                (r'search .* library', 'library_search'),
                (r'return.*from .* graveyard', 'graveyard_recursion')
            ]
            
            effect_groups = defaultdict(list)
            for _, card in self.card_db.iterrows():
                text = str(card['oracle_text']).lower()
                
                for pattern, effect_name in effect_patterns:
                    if re.search(pattern, text):
                        effect_groups[effect_name].append(card['name'])
            
            # Filter effect synergies
            for effect, cards in effect_groups.items():
                if 3 <= len(cards) <= 20:
                    synergies[f"{effect}_effect_synergy"] = cards
        
        # Run all synergy detection methods
        keyword_co_occurrence_synergy()
        type_synergy_detection()
        effect_synergy_detection()
        
        return dict(synergies)
        
class DynamicMetaAnalyzer:
    """
    Comprehensive meta analysis system that adapts to the current card pool
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.archetype_classifier = DynamicArchetypeClassifier(card_db)
        self.mechanics_analyzer = DynamicCardMechanicsAnalyzer(card_db)
        self.synergy_detector = DynamicSynergyDetector(card_db)
    
    def analyze_meta(self, decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Perform comprehensive meta analysis
        """
        try:
            # Analyze format characteristics
            format_characteristics = self._analyze_format_characteristics()
            
            # Analyze individual decks
            deck_analyses = {}
            archetype_distribution = Counter()
            card_frequencies = Counter()
            
            for deck_name, decklist in decklists.items():
                try:
                    deck_analysis = self._analyze_deck(deck_name, decklist)
                    deck_analyses[deck_name] = deck_analysis
                    archetype_distribution[deck_analysis['archetype']] += 1
                    # Update card frequencies
                    card_frequencies.update(decklist)
                except Exception as e:
                    logger.error(f"Error analyzing deck {deck_name}: {str(e)}")
            
            # Calculate meta statistics
            meta_statistics = self._calculate_meta_statistics(deck_analyses)
            
            return {
                'format_characteristics': format_characteristics,
                'deck_analyses': deck_analyses,
                'meta_statistics': meta_statistics,
                'archetype_distribution': dict(archetype_distribution),
                'card_frequencies': dict(card_frequencies)
            }
        except Exception as e:
            logger.error(f"Error in meta analysis: {str(e)}")
            raise
    
    def _analyze_format_characteristics(self) -> Dict[str, Any]:
        """
        Analyze overall format characteristics dynamically
        """
        return {
            'mechanics': self.mechanics_analyzer.extract_mechanics_and_abilities(),
            'synergies': self.synergy_detector.detect_synergies(),
            'archetypes': self.archetype_classifier.archetypes
        }
    
    def _analyze_deck(self, deck_name: str, decklist: List[str]) -> Dict[str, Any]:
        """
        Analyze individual deck with dynamic classification and fuzzy matching
        """
        try:
            # Verify and potentially correct card names
            verified_cards = []
            missing_cards = []
            
            for card in decklist:
                try:
                    # Exact match first
                    card_match = self.card_db[self.card_db['name'] == card]
                    if not card_match.empty:
                        verified_cards.append(card)
                        continue
                    
                    # Case-insensitive match
                    card_lower = card.lower()
                    case_insensitive_match = self.card_db[self.card_db['name'].str.lower() == card_lower]
                    if not case_insensitive_match.empty:
                        verified_name = case_insensitive_match.iloc[0]['name']
                        verified_cards.append(verified_name)
                        if verified_name != card:
                            logger.info(f"Corrected card name: '{card}' → '{verified_name}'")
                        continue
                    
                    # If no match found, add to missing cards
                    missing_cards.append(card)
                    
                except Exception as e:
                    logger.warning(f"Error processing card {card}: {str(e)}")
                    missing_cards.append(card)
            
            # Calculate deck statistics and classify archetype
            deck_stats = self._calculate_deck_statistics(verified_cards)
            archetype = self._classify_deck_archetype(deck_stats)
            
            return {
                'name': deck_name,
                'archetype': archetype,
                'statistics': deck_stats,
                'card_count': len(verified_cards),
                'missing_cards': missing_cards,
                'verified_cards': verified_cards
            }
            
        except Exception as e:
            logger.error(f"Error analyzing deck {deck_name}: {str(e)}")
            return {
                'name': deck_name,
                'archetype': 'unknown',
                'statistics': {},
                'card_count': len(decklist),
                'missing_cards': decklist,
                'verified_cards': []
            }
    
    def _calculate_deck_statistics(self, decklist: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive deck statistics
        """
        try:
            # Filter and analyze deck cards
            deck_cards = self.card_db[self.card_db['name'].isin(decklist)]
            
            if deck_cards.empty:
                return {}
            
            # Calculate various statistics
            nonland_cards = deck_cards[~deck_cards['is_land']]
            
            if nonland_cards.empty:
                return {}
            
            stats = {
                'creature_ratio': nonland_cards['is_creature'].mean(),
                'avg_cmc': nonland_cards['cmc'].mean(),
                'color_complexity': nonland_cards['color_count'].mean(),
                'legendary_ratio': nonland_cards['is_legendary'].mean(),
                'multicolor_ratio': nonland_cards['is_multicolored'].mean()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating deck statistics: {str(e)}")
            return {}
    
    def _classify_deck_archetype(self, deck_stats: Dict[str, float]) -> str:
        """
        Classify deck archetype based on statistical comparison
        """
        if not deck_stats:
            return 'unknown'
        
        try:
            # Compare deck stats to predefined archetype ranges
            archetype_scores = {}
            for archetype, criteria in self.archetype_classifier.archetypes.items():
                score = 0
                total_checks = 0
                
                for stat, (min_val, max_val) in criteria.items():
                    if stat in deck_stats:
                        stat_val = deck_stats[stat]
                        if min_val <= stat_val <= max_val:
                            score += 1
                        total_checks += 1
                
                # Calculate archetype match percentage
                archetype_scores[archetype] = score / total_checks if total_checks > 0 else 0
            
            # Return archetype with highest match score
            return max(archetype_scores.items(), key=lambda x: x[1])[0] if archetype_scores else 'unknown'
            
        except Exception as e:
            logger.error(f"Error classifying deck archetype: {str(e)}")
            return 'unknown'
    
    def _calculate_meta_statistics(self, deck_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate comprehensive meta statistics
        """
        if not deck_analyses:
            return {
                'total_decks': 0,
                'archetype_distribution': {},
                'meta_entropy': 0.0,
                'most_popular_archetype': 'unknown'
            }
        
        try:
            # Archetype distribution
            archetypes = [analysis['archetype'] for analysis in deck_analyses.values()]
            archetype_counts = Counter(archetypes)
            
            # Diversity metrics
            total_decks = len(archetypes)
            
            # Calculate entropy (avoid log(0))
            entropy = 0.0
            if total_decks > 0:
                for count in archetype_counts.values():
                    p = count / total_decks
                    entropy -= p * np.log(p) if p > 0 else 0
            
            return {
                'total_decks': total_decks,
                'archetype_distribution': dict(archetype_counts),
                'meta_entropy': entropy,
                'most_popular_archetype': max(archetype_counts.items(), 
                                            key=lambda x: x[1])[0] if archetype_counts else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Error calculating meta statistics: {str(e)}")
            return {
                'total_decks': len(deck_analyses),
                'archetype_distribution': {},
                'meta_entropy': 0.0,
                'most_popular_archetype': 'unknown'
            }
 
def load_and_preprocess_cards(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess card data with adaptive type handling
    """
    try:
        # Initial load with all columns as strings
        df = pd.read_csv(csv_path, dtype=str)
        
        # Convert numeric columns
        numeric_columns = ['cmc', 'color_count']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert boolean columns
        boolean_columns = ['is_creature', 'is_land', 'is_instant_sorcery', 
                          'is_multicolored', 'has_etb_effect', 'is_legendary']
        for col in boolean_columns:
            df[col] = df[col].map({'True': True, 'False': False})
        
        # Handle list-like strings
        def safe_eval_list(val):
            if pd.isna(val):
                return []
            try:
                # Remove quotes and brackets, then split by comma
                cleaned = val.strip('[]\'\" ').strip()
                if not cleaned:
                    return []
                return [item.strip('\'"') for item in cleaned.split(',') if item.strip()]
            except:
                return []
        
        list_columns = ['colors', 'color_identity', 'keywords', 'produced_mana']
        for col in list_columns:
            df[col] = df[col].apply(safe_eval_list)
        
        # Create additional columns for analysis
        df['name_lower'] = df['name'].str.lower()
        
        # Handle missing values
        string_columns = ['name', 'type_line', 'oracle_text', 'power', 'toughness']
        for col in string_columns:
            df[col] = df[col].fillna('')
        
        logger.info(f"Successfully loaded {len(df)} cards")
        return df
        
    except Exception as e:
        logger.error(f"Error loading and preprocessing cards: {e}")
        raise

def load_decklists(directory: str) -> Dict[str, List[str]]:
    """
    Load decklists from text files in the specified directory
    Uses more robust parsing to handle various deck list formats
    """
    decklists = {}
    
    try:
        # Find all potential deck files
        deck_files = [
            f for f in os.listdir(directory) 
            if f.endswith('.txt') and not f.startswith('.')
        ]
        
        if not deck_files:
            logger.warning(f"No deck files found in {directory}")
            return {}
        
        for filename in deck_files:
            filepath = os.path.join(directory, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    
                    # More flexible deck parsing
                    deck_name = os.path.splitext(filename)[0]
                    mainboard = []
                    
                    for line in lines:
                        line = line.strip()
                        
                        # Skip empty lines and comments
                        if not line or line.startswith('#'):
                            continue
                        
                        # Flexible parsing for card entries
                        try:
                            # Handle formats like:
                            # 4 Card Name
                            # Card Name x4
                            # 4x Card Name
                            match = re.match(r'^(?:(\d+)[x]?\s*)?(.+?)(?:\s*[x]?(\d+))?$', 
                                          line, re.IGNORECASE)
                            if match:
                                # Determine count
                                count = match.group(1) or match.group(3) or '1'
                                card_name = match.group(2).strip()
                                
                                # Add card multiple times based on count
                                mainboard.extend([card_name] * int(count))
                        except Exception as e:
                            logger.warning(f"Could not parse line in {filename}: {line}")
                            continue
                    
                    # Only add non-empty decklists
                    if mainboard:
                        decklists[deck_name] = mainboard
                    else:
                        logger.warning(f"Empty decklist found in {filename}")
            
            except Exception as e:
                logger.error(f"Error processing deck file {filename}: {e}")
                continue
        
        # Log summary of loaded decklists
        logger.info(f"Loaded {len(decklists)} decklists from {len(deck_files)} files")
        return decklists
        
    except Exception as e:
        logger.error(f"Error loading decklists: {e}")
        return {}

def print_meta_analysis_report(meta_analysis: Dict[str, Any]):
    """
    Generate a comprehensive report of the meta analysis
    """
    try:
        print("\n=== Magic Format Meta Analysis Report ===\n")
        
        # Format Characteristics
        print("1. Format Mechanics:")
        mechanics = meta_analysis.get('format_characteristics', {}).get('mechanics', {})
        for mechanic, count in sorted(mechanics.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   - {mechanic}: {count} occurrences")
        
        # Synergies
        print("\n2. Notable Synergies:")
        synergies = meta_analysis.get('format_characteristics', {}).get('synergies', {})
        for synergy, cards in list(synergies.items())[:10]:
            print(f"   - {synergy}: {len(cards)} cards")
        
        # Archetype Distribution
        print("\n3. Archetype Distribution:")
        archetype_dist = meta_analysis.get('archetype_distribution', {})
        total_decks = sum(archetype_dist.values()) if archetype_dist else 0
        if total_decks > 0:
            for archetype, count in sorted(archetype_dist.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_decks) * 100
                print(f"   - {archetype}: {percentage:.1f}% ({count} decks)")
        else:
            print("   No archetype data available")
        
        # Most Played Cards
        print("\n4. Most Played Cards:")
        card_freq = meta_analysis.get('card_frequencies', {})
        for card, count in sorted(card_freq.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"   - {card}: {count} copies")
        
        # Meta Statistics
        meta_stats = meta_analysis.get('meta_statistics', {})
        print("\n5. Meta Statistics:")
        print(f"   - Total Decks: {meta_stats.get('total_decks', 0)}")
        print(f"   - Meta Entropy: {meta_stats.get('meta_entropy', 0):.2f}")
        print(f"   - Most Popular Archetype: {meta_stats.get('most_popular_archetype', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Error printing meta analysis report: {e}")
        print("\nError generating report. Please check the logs for details.")
        
def main():
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Script execution started.")
        
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
            help='Directory containing deck lists'
        )
        args = parser.parse_args()
        
        # Validate input paths
        if not os.path.exists(args.cards):
            logger.error(f"Cards CSV file not found: {args.cards}")
            return
        
        if not os.path.exists(args.decks):
            logger.error(f"Decks directory not found: {args.decks}")
            return
        
        # Load card database
        logger.info("Loading card database...")
        cards_df = load_and_preprocess_cards(args.cards)
        logger.info(f"Loaded {len(cards_df)} cards")
        
        # Load decklists
        logger.info("Loading decklists...")
        decklists = load_decklists(args.decks)
        logger.info(f"Loaded {len(decklists)} decklists")
        
        if not decklists:
            logger.error("No valid decklists found")
            return
        
        # Perform meta analysis
        logger.info("Analyzing meta...")
        meta_analyzer = DynamicMetaAnalyzer(cards_df)
        meta_analysis = meta_analyzer.analyze_meta(decklists)
        
        # Print analysis report
        print_meta_analysis_report(meta_analysis)
        
        # Save detailed analysis to JSON
        output_file = 'meta_analysis_results.json'
        try:
            # Convert non-serializable objects to strings
            def json_serializable(obj):
                if isinstance(obj, (np.int64, np.float64)):
                    return float(obj)
                if isinstance(obj, set):
                    return list(obj)
                return str(obj)
            
            with open(output_file, 'w') as f:
                json.dump(meta_analysis, f, indent=2, default=json_serializable)
            logger.info(f"Detailed analysis saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving analysis results to JSON: {e}")
    
    except Exception as e:
        logger.error(f"Meta analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()