import os
import re
import json
import logging
import itertools
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import Counter, defaultdict
from enum import Enum, auto
from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Attempt to import Levenshtein, provide fallback
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    logger.warning("Levenshtein library not found. Fuzzy matching will be limited.")
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Simple Levenshtein distance fallback"""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

def string_similarity(s1: str, s2: str) -> float:
    """Calculate string similarity using Levenshtein distance"""
    max_len = max(len(s1), len(s2))
    return 1 - (levenshtein_distance(s1, s2) / max_len) if max_len > 0 else 1.0

class DynamicCardMechanicsAnalyzer:
    """
    Advanced card mechanics analyzer that extracts mechanics from keywords
    and identifies non-keyword mechanics from oracle text
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        # Extract all unique keywords from the card pool
        self.keywords = self._extract_all_keywords()
        # Define non-keyword mechanics patterns
        self.non_keyword_mechanics = self._define_non_keyword_mechanics()
        
    def _extract_all_keywords(self) -> Set[str]:
        """Extract all unique keywords from the card pool, case-insensitive"""
        unique_keywords = set()
        for keywords in self.card_db['keywords']:
            if isinstance(keywords, list):
                unique_keywords.update(kw.lower() for kw in keywords)
        return unique_keywords
    
    def _define_non_keyword_mechanics(self) -> Dict[str, str]:
        """
        Define patterns for identifying non-keyword mechanics
        Focuses on historically significant mechanics without keywords
        """
        return {
            'graveyard_recursion': r'return.*from.*graveyard to.*(?:hand|battlefield)',
            'self_mill': r'put.*(?:cards?|top).*library.*graveyard',
            'opponent_mill': r'target.*(?:player|opponent).*puts?.*library.*graveyard',
            'ramp': r'search.*library.*basic land.*battlefield',
            'tutor': r'search.*library.*(?!basic land)',
            'card_draw': r'draw (?:a|[0-9]+) cards?',
            'looting': r'draw.*cards?.*discard.*cards?',
            'burn': r'deals? [0-9]+ damage to (?:any|target)',
            'board_wipe': r'destroy all|deals? [0-9]+ damage to (each|all) creatures?',
            'counter_spell': r'counter target spell',
            'token_generation': r'create.*token',
            'sacrifice_effect': r'sacrifice a',
            'tap_effect': r'tap target',
            'bounce': r'return target.*to.*owners? hand',
            'anthem_effect': r'creatures you control get',
            'mana_generation': r'add \{[WUBRGC]\}',
            'life_gain': r'gain [0-9]+ life',
            'life_loss': r'lose [0-9]+ life',
            'scry_effect': r'scry [0-9]+',
            'surveil': r'surveil [0-9]+'
        }
    
    def extract_mechanics_and_abilities(self) -> Dict[str, int]:
        """Extract all mechanics, combining keywords and non-keyword effects"""
        mechanics = Counter()
        
        # First count keywords (case-insensitive)
        for keywords in self.card_db['keywords']:
            if isinstance(keywords, list):
                mechanics.update(kw.lower() for kw in keywords)
        
        # Then identify non-keyword mechanics from oracle text
        for _, card in self.card_db.iterrows():
            oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
            
            for mechanic_name, pattern in self.non_keyword_mechanics.items():
                # Only add if not already a keyword
                if mechanic_name not in self.keywords and re.search(pattern, oracle_text, re.IGNORECASE):
                    mechanics[mechanic_name] += 1
        
        # Filter out rare mechanics (less than 1% of cards)
        min_threshold = len(self.card_db) * 0.01
        return {k: v for k, v in mechanics.items() if v >= min_threshold}
        
class DynamicSynergyDetector:
    """
    Advanced synergy detector that identifies meaningful card interactions
    based on meta-relevant patterns
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        # Create indices for faster lookups
        self._create_indices()
        # Extract all subtypes from the card pool
        self.subtypes = self._extract_subtypes()
        
    def _create_indices(self):
        """Create indices for efficient card lookups"""
        self.card_index = {
            name: card for name, card in 
            self.card_db[['name', 'type_line', 'oracle_text', 'keywords']].iterrows()
        }
        
    def _extract_subtypes(self) -> Dict[str, Set[str]]:
        """
        Extract all subtypes from the card pool, organized by card type
        """
        subtypes = {
            'Creature': set(),
            'Artifact': set(),
            'Enchantment': set(),
            'Instant': set(),
            'Sorcery': set(),
            'Planeswalker': set()
        }
        
        for _, card in self.card_db.iterrows():
            type_line = str(card['type_line'])
            # Split on em dash to separate types from subtypes
            parts = type_line.split('—')
            
            if len(parts) > 1:
                main_types = set(parts[0].strip().split())
                sub_types = set(parts[1].strip().split())
                
                # Assign subtypes to appropriate categories
                for main_type in main_types:
                    if main_type in subtypes:
                        subtypes[main_type].update(sub_types)
        
        return subtypes
    
    def detect_synergies(self, meta_decks: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Detect synergies focusing on what's actually played in the meta
        """
        synergies = {}
        
        # Get all cards being played in the meta
        meta_cards = set(card for deck in meta_decks.values() for card in deck)
        meta_cards_df = self.card_db[self.card_db['name'].isin(meta_cards)]
        
        # Detect various types of synergies
        keyword_synergies = self._detect_keyword_synergies(meta_cards_df)
        subtype_synergies = self._detect_subtype_synergies(meta_cards_df)
        mechanical_synergies = self._detect_mechanical_synergies(meta_cards_df)
        
        # Combine all synergies
        synergies.update(keyword_synergies)
        synergies.update(subtype_synergies)
        synergies.update(mechanical_synergies)
        
        return synergies
    
    def _detect_keyword_synergies(self, meta_cards_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect synergies based on keywords that appear frequently in meta decks
        """
        synergies = {}
        keyword_counts = Counter()
        keyword_cards = defaultdict(list)
        
        for _, card in meta_cards_df.iterrows():
            if isinstance(card['keywords'], list):
                for keyword in card['keywords']:
                    keyword_lower = keyword.lower()
                    keyword_counts[keyword_lower] += 1
                    keyword_cards[keyword_lower].append(card['name'])
                    
                    # Also check for cards that care about this keyword
                    if keyword_lower in str(card['oracle_text']).lower():
                        keyword_cards[f"{keyword_lower}_payoff"].append(card['name'])
        
        # Only include significant keyword synergies
        min_count = len(meta_cards_df) * 0.1  # At least 10% of meta cards
        for keyword, count in keyword_counts.items():
            if count >= min_count:
                cards = list(set(keyword_cards[keyword]))
                payoff_cards = list(set(keyword_cards.get(f"{keyword}_payoff", [])))
                
                if len(cards) >= 3:
                    synergies[f"{keyword}_synergy"] = cards
                if len(payoff_cards) >= 2:
                    synergies[f"{keyword}_payoff_synergy"] = payoff_cards
        
        return synergies
    
    def _detect_subtype_synergies(self, meta_cards_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect synergies based on card subtypes present in meta decks
        """
        synergies = {}
        
        for card_type, possible_subtypes in self.subtypes.items():
            # Filter cards of this type
            type_cards = meta_cards_df[
                meta_cards_df['type_line'].str.contains(card_type, case=False, na=False)
            ]
            
            if type_cards.empty:
                continue
                
            subtype_counts = Counter()
            subtype_cards = defaultdict(list)
            payoff_cards = defaultdict(list)
            
            for _, card in type_cards.iterrows():
                type_line = str(card['type_line'])
                oracle_text = str(card['oracle_text']).lower()
                
                # Extract subtypes for this card
                if '—' in type_line:
                    subtypes = set(type_line.split('—')[1].strip().split())
                    for subtype in subtypes:
                        subtype_lower = subtype.lower()
                        subtype_counts[subtype_lower] += 1
                        subtype_cards[subtype_lower].append(card['name'])
                
                # Check for payoff cards
                for subtype in possible_subtypes:
                    subtype_lower = subtype.lower()
                    if subtype_lower in oracle_text:
                        payoff_cards[subtype_lower].append(card['name'])
            
            # Add significant subtype synergies
            min_count = len(type_cards) * 0.1  # At least 10% of cards of this type
            for subtype, count in subtype_counts.items():
                if count >= min_count:
                    cards = list(set(subtype_cards[subtype]))
                    payoffs = list(set(payoff_cards[subtype]))
                    
                    if len(cards) >= 3:
                        synergies[f"{subtype}_{card_type.lower()}_synergy"] = cards
                    if len(payoffs) >= 2:
                        synergies[f"{subtype}_{card_type.lower()}_payoff_synergy"] = payoffs
        
        return synergies
    
    def _detect_mechanical_synergies(self, meta_cards_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect synergies based on mechanical interactions
        """
        synergies = {}
        
        # Define mechanical patterns to look for
        mechanical_patterns = {
            'sacrifice': (r'sacrifice a', r'when.*sacrificed|whenever.*sacrifice'),
            'graveyard': (r'from.*graveyard', r'whenever.*enters.*graveyard'),
            'etb': (r'enters the battlefield', r'whenever.*enters'),
            'spell_cast': (r'when you cast', r'whenever.*cast'),
            'counter': (r'counter target', r'whenever.*countered'),
            'tokens': (r'create.*token', r'whenever.*token.*enters'),
            'lifegain': (r'gain.*life', r'whenever.*gain.*life'),
            'discard': (r'discard.*card', r'whenever.*discard'),
            'tap': (r'tap target', r'whenever.*becomes tapped'),
            'attack': (r'attacks', r'whenever.*attacks')
        }
        
        for mechanic, (trigger_pattern, payoff_pattern) in mechanical_patterns.items():
            # Find cards with the mechanic
            trigger_cards = meta_cards_df[
                meta_cards_df['oracle_text'].str.contains(
                    trigger_pattern, case=False, na=False
                )
            ]['name'].tolist()
            
            # Find payoff cards
            payoff_cards = meta_cards_df[
                meta_cards_df['oracle_text'].str.contains(
                    payoff_pattern, case=False, na=False
                )
            ]['name'].tolist()
            
            if len(trigger_cards) >= 3:
                synergies[f"{mechanic}_synergy"] = trigger_cards
            if len(payoff_cards) >= 2:
                synergies[f"{mechanic}_payoff_synergy"] = payoff_cards
        
        return synergies

class DynamicArchetypeClassifier:
    """
    Advanced deck archetype classifier that considers meta speed and hybrid strategies
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.archetype_characteristics = self._define_archetype_characteristics()
        self.meta_speed_indicators = self._calculate_meta_speed_indicators()
        
    def _define_archetype_characteristics(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Define core characteristics of each archetype
        """
        return {
            'aggro': {
                'creature_ratio': (0.4, 0.8),
                'avg_cmc': (1.0, 2.5),
                'early_threat_ratio': (0.3, 1.0),
                'interaction_ratio': (0.0, 0.3),
                'card_advantage_ratio': (0.0, 0.2)
            },
            'midrange': {
                'creature_ratio': (0.3, 0.6),
                'avg_cmc': (2.5, 3.5),
                'early_threat_ratio': (0.2, 0.4),
                'interaction_ratio': (0.2, 0.4),
                'card_advantage_ratio': (0.2, 0.4)
            },
            'control': {
                'creature_ratio': (0.1, 0.4),
                'avg_cmc': (2.5, 4.5),
                'early_threat_ratio': (0.0, 0.2),
                'interaction_ratio': (0.3, 0.7),
                'card_advantage_ratio': (0.3, 0.6)
            },
            'combo': {
                'creature_ratio': (0.1, 0.5),
                'avg_cmc': (2.0, 4.0),
                'early_threat_ratio': (0.1, 0.3),
                'interaction_ratio': (0.2, 0.4),
                'card_advantage_ratio': (0.3, 0.6)
            },
            'tempo': {
                'creature_ratio': (0.3, 0.5),
                'avg_cmc': (1.5, 3.0),
                'early_threat_ratio': (0.2, 0.5),
                'interaction_ratio': (0.2, 0.5),
                'card_advantage_ratio': (0.1, 0.3)
            }
        }
    
    def _calculate_meta_speed_indicators(self) -> Dict[str, float]:
        """
        Calculate baseline meta speed indicators from the card pool
        """
        nonland_cards = self.card_db[~self.card_db['is_land']]
        
        # Calculate early game potential
        early_plays = nonland_cards[nonland_cards['cmc'] <= 2]
        early_threats = early_plays[
            (early_plays['is_creature']) & 
            (early_plays['power'].apply(lambda x: str(x).isdigit() and int(str(x)) >= 2))
        ]
        
        # Calculate interaction speed
        interaction_spells = nonland_cards[
            nonland_cards['oracle_text'].str.contains(
                'counter target|destroy target|exile target|deals? \d+ damage to target',
                case=False, na=False
            )
        ]
        
        return {
            'avg_cmc': nonland_cards['cmc'].mean(),
            'early_threat_density': len(early_threats) / len(nonland_cards),
            'interaction_speed': interaction_spells[interaction_spells['cmc'] <= 2].shape[0] / len(nonland_cards),
            'creature_density': nonland_cards['is_creature'].mean()
        }
    
    def classify_deck_archetype(self, deck_cards: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify deck archetype with consideration for hybrid strategies
        """
        # Calculate deck statistics
        stats = self._calculate_deck_statistics(deck_cards)
        
        # Calculate archetype scores
        archetype_scores = {}
        for archetype, characteristics in self.archetype_characteristics.items():
            score = self._calculate_archetype_score(stats, characteristics)
            archetype_scores[archetype] = score
        
        # Identify primary and secondary archetypes
        sorted_scores = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
        primary_score = sorted_scores[0][1]
        
        # Check for hybrid strategy
        hybrid_threshold = 0.7  # Secondary archetype must be at least 70% as strong
        if len(sorted_scores) > 1 and sorted_scores[1][1] >= primary_score * hybrid_threshold:
            return {
                'primary_archetype': 'hybrid',
                'components': [sorted_scores[0][0], sorted_scores[1][0]],
                'scores': archetype_scores,
                'meta_speed_alignment': self._calculate_meta_speed_alignment(stats),
                'statistics': stats
            }
        
        return {
            'primary_archetype': sorted_scores[0][0],
            'scores': archetype_scores,
            'meta_speed_alignment': self._calculate_meta_speed_alignment(stats),
            'statistics': stats
        }
    
    def _calculate_deck_statistics(self, deck_cards: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive deck statistics
        """
        nonland_cards = deck_cards[~deck_cards['is_land']]
        
        # Early game threats
        early_cards = nonland_cards[nonland_cards['cmc'] <= 2]
        early_threats = early_cards[
            (early_cards['is_creature']) & 
            (early_cards['power'].apply(lambda x: str(x).isdigit() and int(str(x)) >= 2))
        ]
        
        stats = {
            'creature_ratio': nonland_cards['is_creature'].mean(),
            'avg_cmc': nonland_cards['cmc'].mean(),
            'early_threat_ratio': len(early_threats) / len(nonland_cards) if len(nonland_cards) > 0 else 0,
            'interaction_ratio': self._calculate_interaction_ratio(nonland_cards),
            'card_advantage_ratio': self._calculate_card_advantage_ratio(nonland_cards)
        }
        
        return stats
    
    def _calculate_interaction_ratio(self, cards: pd.DataFrame) -> float:
        """Calculate ratio of interactive spells"""
        interaction_patterns = [
            r'counter target',
            r'destroy target',
            r'exile target',
            r'deals? \d+ damage to target',
            r'return target.*to.*hand',
            r'tap target',
            r'-\d+/-\d+'
        ]
        
        interaction_count = cards['oracle_text'].apply(
            lambda text: any(
                re.search(pattern, str(text).lower(), re.IGNORECASE) 
                for pattern in interaction_patterns
            )
        ).sum()
        
        return interaction_count / len(cards) if len(cards) > 0 else 0
    
    def _calculate_card_advantage_ratio(self, cards: pd.DataFrame) -> float:
        """Calculate ratio of card advantage spells"""
        advantage_patterns = [
            r'draw \d+ cards?',
            r'search your library',
            r'look at the top \d+ cards?',
            r'return.*from.*graveyard',
            r'scry \d+',
            r'surveil \d+'
        ]
        
        advantage_count = cards['oracle_text'].apply(
            lambda text: any(
                re.search(pattern, str(text).lower(), re.IGNORECASE) 
                for pattern in advantage_patterns
            )
        ).sum()
        
        return advantage_count / len(cards) if len(cards) > 0 else 0
    
    def _calculate_archetype_score(self, stats: Dict[str, float], 
                                 characteristics: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate how well a deck matches an archetype using fuzzy logic
        """
        total_score = 0
        for metric, (min_val, max_val) in characteristics.items():
            if metric in stats:
                val = stats[metric]
                # Calculate how well the value fits within the range
                if min_val <= val <= max_val:
                    # Higher score for being closer to the middle of the range
                    mid = (min_val + max_val) / 2
                    distance_from_mid = abs(val - mid)
                    max_distance = (max_val - min_val) / 2
                    stat_score = 1 - (distance_from_mid / max_distance)
                    total_score += stat_score
                else:
                    # Partial score for being close to the range
                    distance_to_range = min(abs(val - min_val), abs(val - max_val))
                    range_size = max_val - min_val
                    stat_score = max(0, 1 - (distance_to_range / range_size))
                    total_score += stat_score * 0.5  # Penalty for being outside range
        
        return total_score / len(characteristics)
    
    def _calculate_meta_speed_alignment(self, deck_stats: Dict[str, float]) -> float:
        """
        Calculate how well the deck aligns with current meta speed
        """
        alignment_score = 0
        
        # Compare deck stats to meta indicators
        comparisons = [
            ('avg_cmc', 0.5),  # Weight for average CMC comparison
            ('early_threat_ratio', 1.0),  # Higher weight for early game presence
            ('interaction_ratio', 0.8)  # Weight for interaction speed
        ]
        
        total_weight = sum(weight for _, weight in comparisons)
        
        for stat, weight in comparisons:
            if stat in deck_stats and stat in self.meta_speed_indicators:
                deck_val = deck_stats[stat]
                meta_val = self.meta_speed_indicators[stat]
                
                # Calculate similarity (1 = perfect match, 0 = maximum difference)
                similarity = 1 - min(abs(deck_val - meta_val) / max(deck_val, meta_val), 1)
                alignment_score += similarity * weight
        
        return alignment_score / total_weight

class DynamicMetaAnalyzer:
    """
    Comprehensive meta analysis system that adapts to the current card pool
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        # Create indices for faster lookups
        self._create_indices()
        # Initialize analyzers
        self.mechanics_analyzer = DynamicCardMechanicsAnalyzer(card_db)
        self.archetype_classifier = DynamicArchetypeClassifier(card_db)
        self.synergy_detector = DynamicSynergyDetector(card_db)
    
    def _create_indices(self):
        """Create indices for efficient card lookups"""
        # Name-based indices
        self.name_to_card = dict(zip(self.card_db['name'], 
                                    self.card_db.to_dict('records')))
        self.name_lower_to_card = dict(zip(self.card_db['name'].str.lower(), 
                                          self.card_db.to_dict('records')))
        
        # Create basic land set for filtering
        self.basic_lands = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest'}
    
    def analyze_meta(self, decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Perform comprehensive meta analysis
        """
        try:
            # Convert decklists to card objects for efficient processing
            processed_decklists = self._process_decklists(decklists)
            
            # Calculate meta speed and characteristics
            meta_speed = self._calculate_meta_speed(processed_decklists)
            
            # Analyze format characteristics
            format_characteristics = self._analyze_format_characteristics(processed_decklists)
            
            # Analyze individual decks
            deck_analyses = {}
            archetype_distribution = Counter()
            card_frequencies = Counter()
            
            for deck_name, deck_info in processed_decklists.items():
                try:
                    # Analyze deck
                    deck_analysis = self._analyze_deck(deck_name, deck_info)
                    deck_analyses[deck_name] = deck_analysis
                    
                    # Update distributions
                    if isinstance(deck_analysis['archetype'], dict):
                        if deck_analysis['archetype']['primary_archetype'] == 'hybrid':
                            # Split count between hybrid components
                            for component in deck_analysis['archetype']['components']:
                                archetype_distribution[component] += 0.5
                        else:
                            archetype_distribution[deck_analysis['archetype']
                                                ['primary_archetype']] += 1
                    
                    # Update card frequencies (excluding basic lands)
                    for card, count in deck_info['card_counts'].items():
                        if card not in self.basic_lands:
                            card_frequencies[card] += count
                            
                except Exception as e:
                    logger.error(f"Error analyzing deck {deck_name}: {str(e)}")
            
            # Calculate meta statistics
            meta_statistics = self._calculate_meta_statistics(
                deck_analyses, meta_speed, card_frequencies
            )
            
            return {
                'meta_speed': meta_speed,
                'format_characteristics': format_characteristics,
                'deck_analyses': deck_analyses,
                'meta_statistics': meta_statistics,
                'archetype_distribution': dict(archetype_distribution),
                'card_frequencies': dict(card_frequencies)
            }
            
        except Exception as e:
            logger.error(f"Error in meta analysis: {str(e)}")
            raise
    
    def _process_decklists(self, decklists: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Process decklists into a more efficient format
        """
        processed = {}
        for deck_name, card_list in decklists.items():
            # Count cards
            card_counts = Counter(card_list)
            
            # Get card objects
            cards_df = self.card_db[self.card_db['name'].isin(card_counts.keys())]
            
            processed[deck_name] = {
                'card_counts': card_counts,
                'cards_df': cards_df,
                'missing_cards': set(card_counts.keys()) - set(cards_df['name'])
            }
        
        return processed
    
    def _calculate_meta_speed(self, 
                            processed_decklists: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate meta speed characteristics
        """
        # Aggregate statistics across all decks
        total_cards = 0
        cmc_sum = 0
        early_plays = 0
        interaction_count = 0
        
        for deck_info in processed_decklists.values():
            cards_df = deck_info['cards_df']
            card_counts = deck_info['card_counts']
            
            for _, card in cards_df.iterrows():
                count = card_counts[card['name']]
                total_cards += count
                
                if not card['is_land']:
                    # CMC analysis
                    cmc_sum += card['cmc'] * count
                    
                    # Early play analysis
                    if card['cmc'] <= 2:
                        early_plays += count
                    
                    # Interaction analysis
                    if self._is_interaction_card(card):
                        interaction_count += count
        
        if total_cards == 0:
            return {'speed': 'unknown'}
        
        avg_cmc = cmc_sum / total_cards
        early_game_ratio = early_plays / total_cards
        interaction_ratio = interaction_count / total_cards
        
        # Determine meta speed
        if avg_cmc < 2.5 and early_game_ratio > 0.3:
            speed = 'fast'
        elif avg_cmc > 3.5 or early_game_ratio < 0.2:
            speed = 'slow'
        else:
            speed = 'medium'
        
        return {
            'speed': speed,
            'avg_cmc': avg_cmc,
            'early_game_ratio': early_game_ratio,
            'interaction_ratio': interaction_ratio
        }
    def _is_interaction_card(self, card: pd.Series) -> bool:
        """Check if a card is an interaction card"""
        if pd.isna(card['oracle_text']):
            return False
            
        interaction_patterns = [
            r'counter target',
            r'destroy target',
            r'exile target',
            r'deals? \d+ damage to target',
            r'return target.*to.*hand',
            r'-\d+/-\d+'
        ]
        
        return any(re.search(pattern, str(card['oracle_text']), re.IGNORECASE) 
                  for pattern in interaction_patterns)
    
    def _analyze_format_characteristics(self, 
                                     processed_decklists: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Analyze format characteristics focusing on meta-relevant patterns
        """
        # Extract mechanics from the card pool
        mechanics = self.mechanics_analyzer.extract_mechanics_and_abilities()
        
        # Detect synergies in meta decks
        meta_decks = {
            name: list(info['card_counts'].keys()) 
            for name, info in processed_decklists.items()
        }
        synergies = self.synergy_detector.detect_synergies(meta_decks)
        
        # Get archetype characteristics
        archetypes = self.archetype_classifier.archetype_characteristics
        
        return {
            'mechanics': mechanics,
            'synergies': synergies,
            'archetypes': archetypes
        }
    
    def _analyze_deck(self, deck_name: str, deck_info: Dict) -> Dict[str, Any]:
        """
        Analyze individual deck with comprehensive classification
        """
        try:
            cards_df = deck_info['cards_df']
            missing_cards = deck_info['missing_cards']
            
            # Classify deck archetype
            archetype = self.archetype_classifier.classify_deck_archetype(cards_df)
            
            # Calculate color distribution
            colors = self._calculate_color_distribution(cards_df)
            
            return {
                'name': deck_name,
                'archetype': archetype,
                'colors': colors,
                'card_count': len(cards_df),
                'missing_cards': list(missing_cards),
                'verified_cards': list(cards_df['name'])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing deck {deck_name}: {str(e)}")
            return {
                'name': deck_name,
                'archetype': {'primary_archetype': 'unknown'},
                'colors': {},
                'card_count': 0,
                'missing_cards': [],
                'verified_cards': []
            }
    
    def _calculate_color_distribution(self, cards_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate color distribution in a deck"""
        all_colors = []
        for colors in cards_df['colors']:
            if isinstance(colors, list):
                all_colors.extend(colors)
        
        if not all_colors:
            return {}
        
        color_counts = Counter(all_colors)
        total_colors = len(all_colors)
        
        return {color: count/total_colors for color, count in color_counts.items()}
    
    def _calculate_meta_statistics(self, deck_analyses: Dict[str, Dict],
                                 meta_speed: Dict[str, float],
                                 card_frequencies: Counter) -> Dict[str, Any]:
        """
        Calculate comprehensive meta statistics
        """
        if not deck_analyses:
            return {
                'total_decks': 0,
                'meta_speed': meta_speed,
                'archetype_diversity': 0.0,
                'color_diversity': 0.0,
                'most_played_cards': [],
                'key_cards': []
            }
        
        # Calculate archetype diversity
        archetype_counts = Counter()
        for analysis in deck_analyses.values():
            if isinstance(analysis['archetype'], dict):
                if analysis['archetype']['primary_archetype'] == 'hybrid':
                    for component in analysis['archetype']['components']:
                        archetype_counts[component] += 0.5
                else:
                    archetype_counts[analysis['archetype']['primary_archetype']] += 1
        
        # Calculate entropy for diversity metrics
        total_decks = len(deck_analyses)
        archetype_diversity = self._calculate_entropy(archetype_counts.values(), total_decks)
        
        # Identify key cards (cards that appear in multiple decks)
        key_cards = [
            card for card, count in card_frequencies.items()
            if count >= total_decks * 0.2  # Present in at least 20% of decks
        ]
        
        return {
            'total_decks': total_decks,
            'meta_speed': meta_speed,
            'archetype_diversity': archetype_diversity,
            'most_played_cards': [
                {'card': card, 'count': count}
                for card, count in card_frequencies.most_common(15)
                if card not in self.basic_lands
            ],
            'key_cards': key_cards
        }
    
    def _calculate_entropy(self, counts, total: int) -> float:
        """Calculate Shannon entropy for diversity measurement"""
        if total == 0:
            return 0.0
            
        entropy = 0.0
        for count in counts:
            p = count / total
            if p > 0:
                entropy -= p * np.log(p)
        
        return entropy

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
            if col in df.columns:
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
        
        # 1. Meta Speed and Characteristics
        meta_speed = meta_analysis['meta_statistics']['meta_speed']
        print("1. Meta Speed Analysis:")
        print(f"   - Speed: {meta_speed['speed'].title()}")
        print(f"   - Average CMC: {meta_speed['avg_cmc']:.2f}")
        print(f"   - Early Game Ratio: {meta_speed['early_game_ratio']*100:.1f}%")
        print(f"   - Interaction Ratio: {meta_speed['interaction_ratio']*100:.1f}%")
        
        # 2. Format Mechanics (case-insensitive, deduplicated)
        print("\n2. Format Mechanics:")
        mechanics = meta_analysis.get('format_characteristics', {}).get('mechanics', {})
        # Combine counts for same mechanic with different cases
        normalized_mechanics = {}
        for mechanic, count in mechanics.items():
            mechanic_lower = mechanic.lower()
            normalized_mechanics[mechanic_lower] = (
                normalized_mechanics.get(mechanic_lower, 0) + count
            )
        
        for mechanic, count in sorted(normalized_mechanics.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]:
            print(f"   - {mechanic.title()}: {count} occurrences")
        
        # 3. Meta-Relevant Synergies
        print("\n3. Meta-Relevant Synergies:")
        synergies = meta_analysis.get('format_characteristics', {}).get('synergies', {})
        # Group synergies by type
        synergy_groups = {
            'Tribal': [],
            'Mechanical': [],
            'Keyword': []
        }
        
        for synergy, cards in synergies.items():
            if '_creature_type_synergy' in synergy:
                synergy_groups['Tribal'].append((synergy, len(cards)))
            elif '_keyword_synergy' in synergy:
                synergy_groups['Keyword'].append((synergy, len(cards)))
            else:
                synergy_groups['Mechanical'].append((synergy, len(cards)))
        
        for group_name, group_synergies in synergy_groups.items():
            if group_synergies:
                print(f"\n   {group_name} Synergies:")
                for synergy, card_count in sorted(group_synergies, 
                                                key=lambda x: x[1], reverse=True)[:5]:
                    # Clean up synergy name for display
                    display_name = (synergy.replace('_synergy', '')
                                         .replace('_', ' ')
                                         .title())
                    print(f"   - {display_name}: {card_count} cards")
        
        # 4. Archetype Distribution
        print("\n4. Archetype Distribution:")
        archetype_dist = meta_analysis.get('archetype_distribution', {})
        total_decks = sum(archetype_dist.values())
        
        if total_decks > 0:
            print(f"   Total Decks: {total_decks}")
            print(f"   Archetype Diversity: {meta_analysis['meta_statistics']['archetype_diversity']:.2f}")
            print("\n   Distribution:")
            for archetype, count in sorted(archetype_dist.items(), 
                                         key=lambda x: x[1], reverse=True):
                percentage = (count / total_decks) * 100
                print(f"   - {archetype.title()}: {percentage:.1f}% ({count} decks)")
        else:
            print("   No archetype data available")
        
        # 5. Most Played Cards (excluding basics)
        print("\n5. Most Played Cards:")
        most_played = meta_analysis['meta_statistics']['most_played_cards']
        for card_info in most_played[:15]:
            print(f"   - {card_info['card']}: {card_info['count']} copies")
        
        # 6. Key Meta Cards
        print("\n6. Key Meta Cards (20%+ deck presence):")
        key_cards = meta_analysis['meta_statistics']['key_cards']
        for card in sorted(key_cards):
            print(f"   - {card}")
        
        # 7. Deck Analysis Summary
        print("\n7. Individual Deck Analysis Summary:")
        deck_analyses = meta_analysis.get('deck_analyses', {})
        for deck_name, analysis in deck_analyses.items():
            print(f"\n   {deck_name}:")
            
            # Print archetype information
            archetype_info = analysis['archetype']
            if isinstance(archetype_info, dict):
                if archetype_info['primary_archetype'] == 'hybrid':
                    components = ' / '.join(c.title() for c in archetype_info['components'])
                    print(f"   - Archetype: Hybrid ({components})")
                else:
                    print(f"   - Archetype: {archetype_info['primary_archetype'].title()}")
            
            # Print color distribution
            colors = analysis.get('colors', {})
            if colors:
                color_str = ', '.join(f"{color}: {pct*100:.1f}%" 
                                    for color, pct in colors.items())
                print(f"   - Colors: {color_str}")
            
            # Print card count and missing cards
            print(f"   - Total Cards: {analysis['card_count']}")
            if analysis.get('missing_cards'):
                print(f"   - Missing Cards: {len(analysis['missing_cards'])}")
        
    except Exception as e:
        logger.error(f"Error printing meta analysis report: {e}")
        print("\nError generating report. Please check the logs for details.")
        
def main():
    """
    Main function to run the Magic format meta analysis
    """
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

# Script entry point
if __name__ == "__main__":
    main()