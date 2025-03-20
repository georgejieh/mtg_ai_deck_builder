import os
import re
import json
import logging
import itertools
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import Counter, defaultdict
from enum import Enum
from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalize card text by removing special characters and standardizing spacing"""
    # Remove special characters but keep meaningful punctuation
    text = re.sub(r'[\'\"//—]', '', text)
    # Standardize spacing
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def normalize_keyword(keyword: str) -> str:
    """Normalize keyword text"""
    # Remove special characters and apostrophes
    keyword = re.sub(r'[\'\"//—]', '', keyword)
    # Remove leading/trailing whitespace
    keyword = keyword.strip()
    # Convert to lowercase
    return keyword.lower()

def normalize_card_name(card_name: str) -> str:
    """
    Normalize card name by standardizing single slash to double slash format
    """
    # Check if the card name contains a single slash but not double slash
    if '/' in card_name and '//' not in card_name:
        # Split by the single slash and rejoin with proper format
        parts = card_name.split('/')
        if len(parts) == 2:
            return f"{parts[0].strip()} // {parts[1].strip()}"
    return card_name

class DynamicArchetypeClassifier:
    """Advanced deck archetype classifier that considers meta speed and hybrid strategies"""
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.archetype_characteristics = self._define_archetype_characteristics()
        self.meta_speed_indicators = self._calculate_meta_speed_indicators()
        logger.info("Archetype Classifier initialized with meta speed analysis")
    
    def _define_archetype_characteristics(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Define core characteristics of each archetype with fuzzy boundaries"""
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
        """Calculate baseline meta speed indicators from the card pool"""
        # Use boolean mask with more robust checking to filter non-land cards
        try:
            nonland_mask = ~(self.card_db['is_land'] == True)
            nonland_cards = self.card_db[nonland_mask]
            
            # Check if nonland_cards is empty and handle it
            if len(nonland_cards) == 0:
                logger.warning("No non-land cards found in the card database")
                return {
                    'avg_cmc': 0.0,
                    'early_threat_density': 0.0,
                    'interaction_speed': 0.0,
                    'creature_density': 0.0
                }
            
            # Calculate early game potential
            early_plays = nonland_cards[nonland_cards['cmc'] <= 2]
            early_threats = early_plays[
                (early_plays['is_creature'] == True) & 
                (early_plays['power'].apply(
                    lambda x: str(x).replace('*', '').isdigit() and 
                             int(str(x).replace('*', '')) >= 2 if pd.notna(x) else False
                ))
            ]
            
            # Calculate interaction speed
            interaction_patterns = [
                r'counter target',
                r'destroy target',
                r'exile target',
                r'deals? \d+ damage to target'
            ]
            
            interaction_spells = nonland_cards[
                nonland_cards['oracle_text'].str.contains(
                    '|'.join(interaction_patterns),
                    case=False, na=False, regex=True
                )
            ]
            
            return {
                'avg_cmc': nonland_cards['cmc'].mean(),
                'early_threat_density': len(early_threats) / max(len(nonland_cards), 1),  # Avoid division by zero
                'interaction_speed': interaction_spells[interaction_spells['cmc'] <= 2].shape[0] / max(len(nonland_cards), 1),
                'creature_density': nonland_cards['is_creature'].mean() if len(nonland_cards) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating meta speed indicators: {str(e)}")
            return {
                'avg_cmc': 0.0,
                'early_threat_density': 0.0,
                'interaction_speed': 0.0,
                'creature_density': 0.0
            }
    
    def _calculate_deck_statistics(self, deck_cards: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive deck statistics"""
        if len(deck_cards) == 0:
            return {
                'creature_ratio': 0.0,
                'avg_cmc': 0.0,
                'early_threat_ratio': 0.0,
                'interaction_ratio': 0.0,
                'card_advantage_ratio': 0.0
            }
        
        # Safer boolean filtering
        try:
            nonland_mask = ~(deck_cards['is_land'] == True)
            nonland_cards = deck_cards[nonland_mask]
            
            if len(nonland_cards) == 0:
                return {
                    'creature_ratio': 0.0,
                    'avg_cmc': 0.0,
                    'early_threat_ratio': 0.0,
                    'interaction_ratio': 0.0,
                    'card_advantage_ratio': 0.0
                }
            
            # Calculate early game threats
            early_cards = nonland_cards[nonland_cards['cmc'] <= 2]
            creature_mask = early_cards['is_creature'] == True
            
            early_threats = early_cards[
                creature_mask & 
                (early_cards['power'].apply(
                    lambda x: str(x).replace('*', '').isdigit() and 
                             int(str(x).replace('*', '')) >= 2 if pd.notna(x) else False
                ))
            ]
            
            # Calculate key ratios
            stats = {
                'creature_ratio': nonland_cards['is_creature'].mean() if len(nonland_cards) > 0 else 0.0,
                'avg_cmc': nonland_cards['cmc'].mean(),
                'early_threat_ratio': len(early_threats) / max(len(nonland_cards), 1),  # Avoid division by zero
                'interaction_ratio': self._calculate_interaction_ratio(nonland_cards),
                'card_advantage_ratio': self._calculate_card_advantage_ratio(nonland_cards)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating deck statistics: {str(e)}")
            return {
                'creature_ratio': 0.0,
                'avg_cmc': 0.0,
                'early_threat_ratio': 0.0,
                'interaction_ratio': 0.0,
                'card_advantage_ratio': 0.0
            }
    
    def _calculate_interaction_ratio(self, cards: pd.DataFrame) -> float:
        """Calculate ratio of interactive spells"""
        if len(cards) == 0:
            return 0.0
            
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
            ) if pd.notna(text) else False
        ).sum()
        
        return interaction_count / max(len(cards), 1)  # Avoid division by zero
    
    def _calculate_card_advantage_ratio(self, cards: pd.DataFrame) -> float:
        """Calculate ratio of card advantage spells"""
        if len(cards) == 0:
            return 0.0
            
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
            ) if pd.notna(text) else False
        ).sum()
        
        return advantage_count / max(len(cards), 1)  # Avoid division by zero

    def classify_deck_archetype(self, deck_cards: pd.DataFrame) -> Dict[str, Any]:
        """Classify deck archetype with consideration for hybrid strategies and meta alignment"""
        # Calculate deck statistics
        stats = self._calculate_deck_statistics(deck_cards)
        
        # Calculate archetype scores
        archetype_scores = {}
        for archetype, characteristics in self.archetype_characteristics.items():
            score = self._calculate_archetype_score(stats, characteristics)
            archetype_scores[archetype] = score
        
        # Identify primary and secondary archetypes
        sorted_scores = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_scores:
            return {
                'primary_archetype': 'unknown',
                'scores': {},
                'meta_speed_alignment': 0.0,
                'statistics': stats
            }
            
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
    
    def _calculate_archetype_score(self, stats: Dict[str, float], 
                                 characteristics: Dict[str, Tuple[float, float]]) -> float:
        """Calculate how well a deck matches an archetype using fuzzy logic"""
        if not stats or not characteristics:
            return 0.0
            
        total_score = 0
        total_checks = 0
        
        for metric, (min_val, max_val) in characteristics.items():
            if metric in stats:
                value = stats[metric]
                if min_val <= value <= max_val:
                    # Higher score for being closer to the middle of the range
                    mid = (min_val + max_val) / 2
                    distance_from_mid = abs(value - mid)
                    max_distance = (max_val - min_val) / 2
                    stat_score = 1 - (distance_from_mid / max_distance)
                    total_score += stat_score
                total_checks += 1
                
        return total_score / max(total_checks, 1)  # Avoid division by zero
    
    def _calculate_meta_speed_alignment(self, deck_stats: Dict[str, float]) -> float:
        """Calculate how well the deck aligns with current meta speed"""
        alignment_score = 0
        
        # Compare deck stats to meta indicators
        comparisons = [
            ('avg_cmc', 0.5),  # Weight for average CMC comparison
            ('early_threat_ratio', 1.0),  # Higher weight for early game presence
            ('interaction_ratio', 0.8)  # Weight for interaction speed
        ]
        
        total_weight = sum(weight for _, weight in comparisons)
        
        if total_weight == 0:
            return 0.0
            
        for stat, weight in comparisons:
            if stat in deck_stats and stat in self.meta_speed_indicators:
                deck_val = deck_stats[stat]
                meta_val = self.meta_speed_indicators[stat]
                
                if deck_val == 0 and meta_val == 0:
                    # Both are zero, consider it a perfect match
                    similarity = 1.0
                else:
                    # Calculate similarity (1 = perfect match, 0 = maximum difference)
                    max_val = max(deck_val, meta_val)
                    if max_val > 0:
                        similarity = 1 - min(abs(deck_val - meta_val) / max_val, 1)
                    else:
                        similarity = 1.0  # Both values are 0
                        
                alignment_score += similarity * weight
        
        return alignment_score / total_weight

class DynamicCardMechanicsAnalyzer:
    """Advanced card mechanics analyzer with improved accuracy"""
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.keywords_set = self._extract_keywords_from_dataset()
        self.non_keyword_mechanics = self._define_non_keyword_mechanics()
        logger.info(f"Card Mechanics Analyzer initialized with {len(self.keywords_set)} keywords")

    def _extract_keywords_from_dataset(self) -> Set[str]:
        """Extract and normalize all unique keywords from the dataset"""
        unique_keywords = set()
        for keywords in self.card_db['keywords']:
            if isinstance(keywords, list):
                unique_keywords.update(normalize_keyword(kw) for kw in keywords)
        return unique_keywords

    def _define_non_keyword_mechanics(self) -> Dict[str, str]:
        """Define regex patterns for identifying card mechanics"""
        return {
            'card_draw': r'draw (?:a|[0-9]+) cards?',
            'tutor': r'search your library for (?:a|an|[0-9]+)',
            'removal': r'(?:destroy|exile) target',
            'counter_spell': r'counter target (?:spell|ability)',
            'bounce': r'return target .*? to (?:its owner\'?s?|their) hand',
            'ramp': r'search your library for (?:a|an) .* land card',
            'board_wipe': r'destroy all|exile all',
            'discard': r'target .*? discards?',
            'life_gain': r'gain [0-9]+ life',
            'damage': r'deals? [0-9]+ damage',
            'pump': r'gets? \+[0-9]+/\+[0-9]+',
            'token_generation': r'create[s]? (?:a|an|[0-9]+|X) .* token',
            'sacrifice_outlet': r'sacrifice a creature:',
            'etb_trigger': r'when .* enters the battlefield',
            'dies_trigger': r'when .* dies',
            'attack_trigger': r'when .* attacks',
            'scry': r'scry [0-9]+',
            'surveil': r'surveil [0-9]+',
            'mill': r'put the top.*card.*of.*library into.*graveyard'
        }

    def analyze_mechanics(self, card: pd.Series) -> Dict[str, bool]:
        """Analyze a single card's mechanics comprehensively"""
        mechanics = {}
        
        # Add keyword mechanics
        if isinstance(card['keywords'], list):
            for keyword in card['keywords']:
                keyword = normalize_keyword(keyword)
                if keyword:
                    mechanics[keyword] = True

        # Check oracle text for mechanical patterns
        oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
        
        # For split cards, try to get the back face oracle text
        if pd.notna(card['full_name']) and ' // ' in card['full_name']:
            front_face, back_face = card['full_name'].split(' // ', 1)
            # If this card is the front face, try to find the back face oracle text
            if card['name'] == front_face:
                back_face_data = self.card_db[self.card_db['name'] == back_face]
                if not back_face_data.empty:
                    back_oracle = back_face_data['oracle_text'].iloc[0]
                    if pd.notna(back_oracle):
                        oracle_text += ' ' + str(back_oracle).lower()
        
        for mechanic_name, pattern in self.non_keyword_mechanics.items():
            if re.search(pattern, oracle_text, re.IGNORECASE):
                mechanics[mechanic_name] = True

        # Add card type based mechanics
        if card['is_creature'] == True:
            mechanics['creature'] = True
            # Add power-based classifications
            try:
                power = float(card['power']) if pd.notna(card['power']) else 0
                if power >= 4:
                    mechanics['big_creature'] = True
                if power <= 2:
                    mechanics['small_creature'] = True
            except (ValueError, TypeError):
                pass
        
        # Special handling for Room cards
        if pd.notna(card['type_line']) and 'Room' in card['type_line']:
            mechanics['room'] = True
            
            if 'unlock' in oracle_text:
                mechanics['unlock_mechanic'] = True
            
            if 'door' in oracle_text:
                mechanics['door_mechanic'] = True

        return mechanics

    def extract_mechanics_and_abilities(self) -> Dict[str, int]:
        """Extract all mechanics and their frequencies from the card pool"""
        mechanics_count = Counter()
        
        for _, card in self.card_db.iterrows():
            card_mechanics = self.analyze_mechanics(card)
            mechanics_count.update(card_mechanics.keys())

        # Filter out very rare mechanics
        min_threshold = max(len(self.card_db) * 0.01, 1)  # At least 1% of cards or 1 card
        significant_mechanics = {
            k: v for k, v in mechanics_count.items() 
            if v >= min_threshold
        }

        return dict(significant_mechanics)

class DynamicSynergyDetector:
    """Improved synergy detection with focus on meaningful interactions including Room cards"""
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.mechanics_analyzer = DynamicCardMechanicsAnalyzer(card_db)
        self.creature_types = self._extract_creature_types()
        self.room_synergies = self._identify_room_synergies()
        self.synergy_patterns = self._define_synergy_patterns()
        
    def _extract_creature_types(self) -> Set[str]:
        """Extract all creature types from the dataset"""
        creature_types = set()
        for _, card in self.card_db.iterrows():
            if card['is_creature'] == True and '—' in str(card['type_line']):
                type_parts = card['type_line'].split('—', 1)
                if len(type_parts) > 1:
                    subtypes = type_parts[1]
                    creature_types.update(
                        normalize_text(subtype) 
                        for subtype in subtypes.strip().split()
                    )
        return creature_types

    def _identify_room_synergies(self) -> Dict[str, List[str]]:
        """Identify Room-type enchantment synergies"""
        room_cards = self.card_db[self.card_db['type_line'].str.contains('Room', na=False)]
        
        # Group room cards by their effects
        unlock_payoffs = []  # Cards that benefit from unlocking doors
        graveyard_synergy = []  # Room cards with graveyard interaction
        token_generators = []  # Room cards that create tokens
        
        for _, card in room_cards.iterrows():
            oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
            
            if 'unlock' in oracle_text:
                unlock_payoffs.append(card['name'])
                
            if any(term in oracle_text for term in ['graveyard', 'dies', 'died']):
                graveyard_synergy.append(card['name'])
                
            if 'create' in oracle_text and 'token' in oracle_text:
                token_generators.append(card['name'])
        
        return {
            'unlock_payoffs': unlock_payoffs,
            'graveyard_room_synergy': graveyard_synergy,
            'token_generating_rooms': token_generators
        }

    def _define_synergy_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Define patterns for identifying synergistic relationships"""
        # Add room-specific patterns
        room_patterns = {
            'room': {
                'enabler_patterns': [
                    r'room',
                    r'unlock',
                    r'door'
                ],
                'payoff_patterns': [
                    r'when.*unlock',
                    r'whenever.*unlock',
                    r'when.*door',
                    r'whenever.*door',
                    r'rooms? you control'
                ]
            }
        }
        
        # Combine with standard patterns
        standard_patterns = {
            'tribal': {
                'enabler_patterns': [
                    r'is a {pattern_key}',
                    r'is a {pattern_key} creature'
                ],
                'payoff_patterns': [
                    r'{pattern_key}s? you control get',
                    r'other {pattern_key}s? you control',
                    r'whenever a {pattern_key} you control',
                    r'for each {pattern_key} you control'
                ]
            },
            'keyword': {
                'enabler_patterns': [
                    r'has {pattern_key}',
                    r'gains? {pattern_key}'
                ],
                'payoff_patterns': [
                    r'creatures? with {pattern_key} get',
                    r'whenever a creature with {pattern_key}',
                    r'for each creature with {pattern_key}'
                ]
            },
            'mechanic': {
                'enabler_patterns': [
                    r'{pattern_key}',
                    r'you may {pattern_key}'
                ],
                'payoff_patterns': [
                    r'whenever you {pattern_key}',
                    r'when.* {pattern_key}',
                    r'if you.* {pattern_key}'
                ]
            }
        }
        
        return {**standard_patterns, **room_patterns}

    def _is_enabler(self, card: pd.Series, synergy_type: str, pattern_key: str) -> bool:
        """Check if a card enables a specific synergy"""
        oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
        
        # Special handling for Room cards
        if synergy_type == 'room':
            # Check if it's a Room card
            if pd.notna(card['type_line']) and 'Room' in card['type_line']:
                return True
                
        # Check if card naturally has the mechanic/keyword
        if synergy_type == 'keyword':
            if isinstance(card['keywords'], list):
                if pattern_key in [normalize_keyword(k) for k in card['keywords']]:
                    return True
                    
        # Check for enabling patterns in text
        patterns = self.synergy_patterns[synergy_type]['enabler_patterns']
        return any(
            re.search(
                pattern.format(pattern_key=pattern_key).lower(), 
                oracle_text,
                re.IGNORECASE
            )
            for pattern in patterns
        )

    def _is_payoff(self, card: pd.Series, synergy_type: str, pattern_key: str) -> bool:
        """Check if a card is a payoff for a specific synergy"""
        oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
        
        # Special handling for Room cards
        if synergy_type == 'room':
            # Check for unlock triggers
            if 'unlock' in oracle_text and ('when' in oracle_text or 'whenever' in oracle_text):
                return True
        
        patterns = self.synergy_patterns[synergy_type]['payoff_patterns']
        return any(
            re.search(
                pattern.format(pattern_key=pattern_key).lower(),
                oracle_text,
                re.IGNORECASE
            )
            for pattern in patterns
        )

    def detect_synergies(self, meta_cards_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Detect meaningful synergies in the card pool including Room synergies"""
        synergies = {
            'tribal': {},
            'keyword': {},
            'mechanical': {},
            'room': {}  # New category for Room synergies
        }
        
        # Handle empty dataframe
        if len(meta_cards_df) == 0:
            return synergies
            
        # Include pre-identified Room synergies
        for synergy_type, cards in self.room_synergies.items():
            filtered_cards = [card for card in cards if card in meta_cards_df['name'].values]
            if filtered_cards:
                synergies['room'][synergy_type] = filtered_cards
                
        # Detect Room mechanic synergies
        room_enablers = []
        room_payoffs = []
        
        for _, card in meta_cards_df.iterrows():
            if self._is_enabler(card, 'room', 'room'):
                room_enablers.append(card['name'])
            if self._is_payoff(card, 'room', 'room'):
                room_payoffs.append(card['name'])
        
        if room_enablers and room_payoffs:
            synergies['room']['room_unlock_synergy'] = {
                'enablers': room_enablers,
                'payoffs': room_payoffs
            }
            
        # Detect tribal synergies
        for creature_type in self.creature_types:
            enablers = []
            payoffs = []
            
            for _, card in meta_cards_df.iterrows():
                if self._is_enabler(card, 'tribal', creature_type):
                    enablers.append(card['name'])
                if self._is_payoff(card, 'tribal', creature_type):
                    payoffs.append(card['name'])
            
            if enablers and payoffs:
                synergies['tribal'][f"{creature_type}_synergy"] = {
                    'enablers': enablers,
                    'payoffs': payoffs
                }
        
        # Detect keyword synergies
        for keyword in self.mechanics_analyzer.keywords_set:
            enablers = []
            payoffs = []
            
            for _, card in meta_cards_df.iterrows():
                if self._is_enabler(card, 'keyword', keyword):
                    enablers.append(card['name'])
                if self._is_payoff(card, 'keyword', keyword):
                    payoffs.append(card['name'])
            
            if enablers and payoffs:
                synergies['keyword'][f"{keyword}_synergy"] = {
                    'enablers': enablers,
                    'payoffs': payoffs
                }

        # Detect mechanical synergies
        for mechanic in self.mechanics_analyzer.non_keyword_mechanics:
            enablers = []
            payoffs = []
            
            for _, card in meta_cards_df.iterrows():
                if self._is_enabler(card, 'mechanic', mechanic):
                    enablers.append(card['name'])
                if self._is_payoff(card, 'mechanic', mechanic):
                    payoffs.append(card['name'])
            
            if enablers and payoffs:
                synergies['mechanical'][f"{mechanic}_synergy"] = {
                    'enablers': enablers,
                    'payoffs': payoffs
                }
        
        return synergies
 
class DynamicMetaAnalyzer:
    """Comprehensive meta analysis system that adapts to the current card pool"""
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        # Create indices for faster lookups
        self._create_indices()
        # Initialize analyzers
        self.mechanics_analyzer = DynamicCardMechanicsAnalyzer(card_db)
        self.archetype_classifier = DynamicArchetypeClassifier(card_db)
        self.synergy_detector = DynamicSynergyDetector(card_db)
        logger.info("Meta Analyzer initialized with all components")
    
    def _create_indices(self):
        """Create indices for efficient card lookups"""
        # Name-based indices
        self.name_to_card = dict(zip(self.card_db['name'], 
                                    self.card_db.to_dict('records')))
        self.name_lower_to_card = dict(zip(self.card_db['name'].str.lower(), 
                                          self.card_db.to_dict('records')))
        
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
        """Perform comprehensive meta analysis"""
        try:
            # Convert decklists to card objects for efficient processing
            processed_decklists = self._process_decklists(decklists)
            
            # Calculate meta speed and characteristics
            meta_speed = self._calculate_meta_speed(processed_decklists)
            logger.info(f"Meta speed calculated as {meta_speed['speed']}")
            
            # Analyze format characteristics
            format_characteristics = self._analyze_format_characteristics(processed_decklists)
            logger.info("Format characteristics analyzed")
            
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
            
            logger.info("Meta analysis completed successfully")
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
        """Process decklists into a more efficient format with improved handling of dual-faced cards"""
        processed = {}
        for deck_name, card_list in decklists.items():
            # Normalize card names in the decklist (convert single slash to double slash)
            normalized_card_list = [normalize_card_name(card) for card in card_list]
            
            # Count cards
            card_counts = Counter(normalized_card_list)
            
            # Match cards in database
            matched_cards_df = self._match_cards_in_database(card_counts.keys())
            
            # Calculate missing cards by comparing matched card names, full names, and front faces
            missing_cards = set()
            matched_names = set(matched_cards_df['name'])
            matched_full_names = set()
            
            for _, card in matched_cards_df.iterrows():
                if pd.notna(card['full_name']):
                    matched_full_names.add(card['full_name'])
            
            for card_name in card_counts.keys():
                # Check if the card is matched by name, full name, or front face
                if (card_name not in matched_names and 
                    card_name not in matched_full_names and
                    not any(card_name.startswith(name + ' // ') for name in matched_names)):
                    missing_cards.add(card_name)
            
            processed[deck_name] = {
                'card_counts': card_counts,
                'cards_df': matched_cards_df,
                'missing_cards': missing_cards
            }
            
            if missing_cards:
                logger.warning(f"Deck {deck_name} has {len(missing_cards)} unrecognized cards: {missing_cards}")
        
        return processed
    
    def _match_cards_in_database(self, card_names: set) -> pd.DataFrame:
        """Match card names to database entries with improved handling of dual-faced cards"""
        # Create an empty DataFrame to store matched cards
        matched_cards = pd.DataFrame()
        remaining_cards = set(card_names)
        
        # First try exact name match
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
                
        # For each remaining card, check if it might be a room card with front face in the db
        if remaining_cards:
            for card_name in list(remaining_cards):
                # Check if this is a room/split card with '//' format
                if '//' in card_name:
                    front_face = card_name.split(' // ')[0].strip()
                    
                    # Try to find the front face in the database
                    front_face_matches = self.card_db[self.card_db['name'] == front_face]
                    if not front_face_matches.empty:
                        # For each matched front face, check if it has a matching full_name
                        for _, card in front_face_matches.iterrows():
                            if pd.notna(card['full_name']) and '//' in card['full_name']:
                                matched_cards = pd.concat([matched_cards, front_face_matches])
                                remaining_cards.remove(card_name)
                                break
                                
                    # If still not matched, try to find any full_name that starts with the front face
                    if card_name in remaining_cards:
                        front_in_full_matches = self.card_db[
                            self.card_db['full_name'].str.startswith(front_face + ' //', na=False)
                        ]
                        if not front_in_full_matches.empty:
                            matched_cards = pd.concat([matched_cards, front_in_full_matches])
                            remaining_cards.remove(card_name)
                
                # Check if this might be a single-faced version of a split card
                elif card_name in self.front_face_to_full_name:
                    full_name = self.front_face_to_full_name[card_name]
                    full_card_matches = self.card_db[self.card_db['full_name'] == full_name]
                    if not full_card_matches.empty:
                        matched_cards = pd.concat([matched_cards, full_card_matches])
                        remaining_cards.remove(card_name)
        
        # Return all matched cards
        return matched_cards
    
    def _calculate_meta_speed(self, processed_decklists: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate meta speed characteristics"""
        total_nonland_cards = 0
        cmc_sum = 0
        early_plays = 0
        interaction_count = 0
        
        for deck_info in processed_decklists.values():
            cards_df = deck_info['cards_df']
            card_counts = deck_info['card_counts']
            
            # First filter nonland cards
            nonland_cards = cards_df[cards_df['is_land'] != True]
            
            for _, card in nonland_cards.iterrows():
                # Get the count based on the card name, handling split cards
                count = 0
                card_name = card['name']
                
                # Try to match by name or full_name
                if card_name in card_counts:
                    count = card_counts[card_name]
                elif pd.notna(card['full_name']) and card['full_name'] in card_counts:
                    count = card_counts[card['full_name']]
                elif pd.notna(card['full_name']) and '//' in card['full_name']:
                    # Check if this is a split card's front face
                    front_face, back_face = card['full_name'].split(' // ', 1)
                    if front_face == card_name and normalize_card_name(front_face) in card_counts:
                        count = card_counts[normalize_card_name(front_face)]
                    # Or check for the full normalized name
                    elif normalize_card_name(card['full_name']) in card_counts:
                        count = card_counts[normalize_card_name(card['full_name'])]
                
                if count > 0:
                    total_nonland_cards += count
                    cmc_sum += card['cmc'] * count
                    
                    # Early play analysis
                    if card['cmc'] <= 2:
                        early_plays += count
                    
                    # Interaction analysis
                    if self._is_interaction_card(card):
                        interaction_count += count
        
        if total_nonland_cards == 0:
            return {
                'speed': 'unknown',
                'avg_cmc': 0.0,
                'early_game_ratio': 0.0,
                'interaction_ratio': 0.0
            }
        
        avg_cmc = cmc_sum / total_nonland_cards
        early_game_ratio = early_plays / total_nonland_cards
        interaction_ratio = interaction_count / total_nonland_cards
        
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
        
        oracle_text = str(card['oracle_text']).lower()
        
        # For split cards, check the back face as well
        if pd.notna(card['full_name']) and ' // ' in card['full_name']:
            front_face, back_face = card['full_name'].split(' // ', 1)
            if card['name'] == front_face:
                back_card = self.card_db[self.card_db['name'] == back_face]
                if not back_card.empty and pd.notna(back_card['oracle_text'].iloc[0]):
                    oracle_text += ' ' + str(back_card['oracle_text'].iloc[0]).lower()
        
        return any(re.search(pattern, oracle_text, re.IGNORECASE) 
                  for pattern in interaction_patterns)

    def _analyze_format_characteristics(self, processed_decklists: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze format characteristics focusing on meta-relevant patterns"""
        # Extract mechanics from the card pool
        mechanics = self.mechanics_analyzer.extract_mechanics_and_abilities()
        
        # Get all cards being played
        meta_cards = set()
        for deck_info in processed_decklists.values():
            meta_cards.update(deck_info['cards_df']['name'])
            
        meta_cards_df = self.card_db[self.card_db['name'].isin(meta_cards)]
        
        # Detect synergies in meta decks
        meta_decks = {
            name: list(info['card_counts'].keys()) 
            for name, info in processed_decklists.items()
        }
        synergies = self.synergy_detector.detect_synergies(meta_cards_df)
        
        # Get archetype characteristics
        archetypes = self.archetype_classifier.archetype_characteristics
        
        return {
            'mechanics': mechanics,
            'synergies': synergies,
            'archetypes': archetypes
        }

    def _analyze_deck(self, deck_name: str, deck_info: Dict) -> Dict[str, Any]:
        """Analyze individual deck with comprehensive classification"""
        try:
            cards_df = deck_info['cards_df']
            missing_cards = deck_info['missing_cards']
            
            # Classify deck archetype
            archetype = self.archetype_classifier.classify_deck_archetype(cards_df)
            
            # Calculate color distribution
            colors = self._calculate_color_distribution(cards_df)
            
            # Get deck's mechanics
            deck_mechanics = self._get_deck_mechanics(deck_info)
            
            return {
                'name': deck_name,
                'archetype': archetype,
                'colors': colors,
                'mechanics': deck_mechanics,
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
                'mechanics': {},
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
    
    def _get_deck_mechanics(self, deck_info: Dict) -> Dict[str, int]:
        """Get mechanics present in the deck with their frequencies"""
        mechanics = Counter()
        
        for _, card in deck_info['cards_df'].iterrows():
            # Get the correct count for the card, handling split cards
            count = 0
            card_name = card['name']
            
            # Try different ways to match the card name
            if card_name in deck_info['card_counts']:
                count = deck_info['card_counts'][card_name]
            elif pd.notna(card['full_name']) and card['full_name'] in deck_info['card_counts']:
                count = deck_info['card_counts'][card['full_name']]
            elif pd.notna(card['full_name']) and '//' in card['full_name']:
                # Check if this is a split card's front face
                front_face, back_face = card['full_name'].split(' // ', 1)
                if front_face == card_name and normalize_card_name(front_face) in deck_info['card_counts']:
                    count = deck_info['card_counts'][normalize_card_name(front_face)]
                # Or check for the full normalized name
                elif normalize_card_name(card['full_name']) in deck_info['card_counts']:
                    count = deck_info['card_counts'][normalize_card_name(card['full_name'])]
            
            if count <= 0:
                continue
            
            # Add keywords
            if isinstance(card['keywords'], list):
                for keyword in card['keywords']:
                    mechanics[keyword.lower()] += count
            
            # Add non-keyword mechanics from oracle text
            oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
            
            # For split cards, include oracle text from both sides
            if pd.notna(card['full_name']) and ' // ' in card['full_name']:
                front_face, back_face = card['full_name'].split(' // ', 1)
                if card['name'] == front_face:
                    back_card = self.card_db[self.card_db['name'] == back_face]
                    if not back_card.empty and pd.notna(back_card['oracle_text'].iloc[0]):
                        oracle_text += ' ' + str(back_card['oracle_text'].iloc[0]).lower()
            
            # Check for room mechanics
            if pd.notna(card['type_line']) and 'Room' in card['type_line']:
                mechanics['room'] += count
                
                if 'unlock' in oracle_text:
                    mechanics['unlock_mechanic'] += count
                
                if 'door' in oracle_text:
                    mechanics['door_mechanic'] += count
            
            # Check for other mechanics
            for mechanic_name, pattern in self.mechanics_analyzer.non_keyword_mechanics.items():
                if re.search(pattern, oracle_text, re.IGNORECASE):
                    mechanics[mechanic_name] += count
        
        return dict(mechanics)
    
    def _calculate_meta_statistics(self, deck_analyses: Dict[str, Dict],
                                 meta_speed: Dict[str, float],
                                 card_frequencies: Counter) -> Dict[str, Any]:
        """Calculate comprehensive meta statistics"""
        if not deck_analyses:
            return {
                'total_decks': 0,
                'meta_speed': meta_speed,
                'most_played_cards': [],
                'key_cards': []
            }
        
        # Calculate most played cards
        most_played = [
            {'card': card, 'count': count}
            for card, count in card_frequencies.most_common(15)
            if card not in self.basic_lands
        ]
        
        # Identify key cards (cards that appear in multiple decks)
        total_decks = len(deck_analyses)
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
    """Load and preprocess card data with improved error handling and split card support"""
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
        
        # Create normalized columns for analysis
        df['name_lower'] = df['name'].str.lower()
        
        # Special handling for Room-type enchantments
        # Make sure they are properly categorized
        df.loc[df['type_line'].str.contains('Room', na=False), 'is_land'] = False
        
        # Add a new column to identify Room cards for easier filtering
        df['is_room'] = df['type_line'].str.contains('Room', na=False)
        
        # Add a column for split cards to assist in identification
        df['is_split'] = df['full_name'].str.contains(' // ', na=False)
        
        # Log counts of special card types
        room_count = df['is_room'].sum() if 'is_room' in df.columns else 0
        split_count = df['is_split'].sum() if 'is_split' in df.columns else 0
        logger.info(f"Successfully loaded {len(df)} cards including {room_count} Room cards and {split_count} split cards")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading and preprocessing cards: {e}")
        raise

def safe_eval_list(val: Any) -> List:
    """Safely evaluate string representations of lists with improved error handling"""
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

def load_decklists(directory: str) -> Dict[str, List[str]]:
    """Load decklists with improved format handling and dual-faced card support"""
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
                                
                                # Do not normalize card names here - leave them in original form
                                # The normalization will happen in the processing stage
                                
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
    """Print comprehensive meta analysis report"""
    print("\n=== Magic Format Meta Analysis Report ===\n")
    
    # Meta Speed
    speed_info = meta_analysis['meta_speed']
    print("1. Meta Speed:")
    print(f"   Speed: {speed_info['speed'].title()}")
    print(f"   Average CMC: {speed_info['avg_cmc']:.2f}")
    print(f"   Early Game Ratio: {speed_info['early_game_ratio']*100:.1f}%")
    print(f"   Interaction Ratio: {speed_info['interaction_ratio']*100:.1f}%")
    
    # Format Mechanics
    print("\n2. Key Mechanics:")
    mechanics = meta_analysis['format_characteristics']['mechanics']
    sorted_mechanics = sorted(mechanics.items(), key=lambda x: x[1], reverse=True)
    for mechanic, count in sorted_mechanics[:15]:
        # Skip mechanics with special characters
        if not any(char in mechanic for char in "'\"//—"):
            print(f"   {mechanic.replace('_', ' ').title()}: {count}")
    
    # Print Room mechanics specifically
    room_mechanics = [m for m in sorted_mechanics if m[0] in ('room', 'unlock_mechanic', 'door_mechanic')]
    if room_mechanics:
        print("\n   Room Mechanics:")
        for mechanic, count in room_mechanics:
            print(f"   {mechanic.replace('_', ' ').title()}: {count}")
    
    # Meaningful Synergies
    print("\n3. Notable Synergies:")
    synergies = meta_analysis['format_characteristics']['synergies']
    
    # Print Room synergies first
    if 'room' in synergies and synergies['room']:
        print("\n   Room Synergies:")
        for synergy_name, data in synergies['room'].items():
            if isinstance(data, dict):
                print(f"   {synergy_name.replace('_', ' ').title()}:")
                print(f"      Enablers: {len(data['enablers'])} cards")
                print(f"      Payoffs: {len(data['payoffs'])} cards")
            else:
                print(f"   {synergy_name.replace('_', ' ').title()}: {len(data)} cards")
    
    # Other synergies
    for category in ['Tribal', 'Keyword', 'Mechanical']:
        if category.lower() in synergies and synergies[category.lower()]:
            print(f"\n   {category} Synergies:")
            for synergy_name, data in synergies[category.lower()].items():
                if isinstance(data, dict):
                    print(f"   {synergy_name.replace('_', ' ').title()}:")
                    print(f"      Enablers: {len(data['enablers'])} cards")
                    print(f"      Payoffs: {len(data['payoffs'])} cards")
    
    # Archetype Distribution
    print("\n4. Archetype Distribution:")
    arch_dist = meta_analysis['archetype_distribution']
    total_decks = sum(arch_dist.values())
    for archetype, count in sorted(arch_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_decks) * 100
        print(f"   {archetype.title()}: {percentage:.1f}% ({count} decks)")
    
    # Most Played Cards
    print("\n5. Most Played Cards:")
    for card_info in meta_analysis['meta_statistics']['most_played_cards'][:15]:
        print(f"   {card_info['card']}: {card_info['count']} copies")
        
    # Print any missing cards that were identified
    missing_cards_count = 0
    for deck_analysis in meta_analysis['deck_analyses'].values():
        missing_cards_count += len(deck_analysis['missing_cards'])
    
    if missing_cards_count > 0:
        print(f"\n6. Warning: {missing_cards_count} card(s) not found in the database")
        print("   Check individual deck analyses for details")

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
            default='json_outputs/parse_meta_analysis_results.json',
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
        cards_df = load_and_preprocess_cards(args.cards)
        
        logger.info("Loading decklists...")
        decklists = load_decklists(args.decks)
        
        if not decklists:
            logger.error("No valid decklists found")
            return
        
        # Perform analysis
        logger.info("Analyzing meta...")
        analyzer = DynamicMetaAnalyzer(cards_df)
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