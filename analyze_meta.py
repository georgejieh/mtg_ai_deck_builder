import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import Counter, defaultdict
from enum import Enum
import re
from dataclasses import dataclass
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArchetypeCharacteristics:
    """Characteristics that define an archetype"""
    creature_ratio: Tuple[float, float]  # min, max
    removal_ratio: Tuple[float, float]
    curve_peak: Tuple[int, int]  # min, max CMC
    interaction_ratio: Tuple[float, float]
    card_advantage_ratio: Tuple[float, float]
    avg_cmc: Tuple[float, float]

class DeckArchetype(Enum):
    AGGRO = "aggro"
    MIDRANGE = "midrange"
    CONTROL = "control"
    TEMPO = "tempo"
    COMBO = "combo"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"

    @classmethod
    def get_characteristics(cls) -> Dict['DeckArchetype', ArchetypeCharacteristics]:
        return {
            cls.AGGRO: ArchetypeCharacteristics(
                creature_ratio=(0.45, 0.75),
                removal_ratio=(0.1, 0.3),
                curve_peak=(1, 2),
                interaction_ratio=(0.0, 0.2),
                card_advantage_ratio=(0.0, 0.15),
                avg_cmc=(1.5, 2.5)
            ),
            cls.MIDRANGE: ArchetypeCharacteristics(
                creature_ratio=(0.35, 0.55),
                removal_ratio=(0.15, 0.35),
                curve_peak=(2, 4),
                interaction_ratio=(0.1, 0.3),
                card_advantage_ratio=(0.1, 0.3),
                avg_cmc=(2.5, 3.5)
            ),
            cls.CONTROL: ArchetypeCharacteristics(
                creature_ratio=(0.1, 0.3),
                removal_ratio=(0.25, 0.45),
                curve_peak=(2, 5),
                interaction_ratio=(0.25, 0.5),
                card_advantage_ratio=(0.2, 0.4),
                avg_cmc=(3.0, 4.5)
            ),
            cls.TEMPO: ArchetypeCharacteristics(
                creature_ratio=(0.3, 0.5),
                removal_ratio=(0.15, 0.35),
                curve_peak=(1, 3),
                interaction_ratio=(0.2, 0.4),
                card_advantage_ratio=(0.1, 0.25),
                avg_cmc=(2.0, 3.0)
            )
        }

class CardCategory(Enum):
    LAND = "land"
    CREATURE = "creature"
    REMOVAL = "removal"
    CARD_ADVANTAGE = "card_advantage"
    INTERACTION = "interaction"
    UTILITY = "utility"
    RAMP = "ramp"
    FINISHER = "finisher"

def safe_parse_list(list_str: str) -> List[str]:
    """Safely parse a string representation of a list"""
    if not list_str or not isinstance(list_str, str):
        return []
    try:
        if list_str.startswith('[') and list_str.endswith(']'):
            # Remove brackets and split by comma
            items = list_str[1:-1].split(',')
            # Clean up each item
            return [item.strip().strip('"\'') for item in items if item.strip()]
        return []
    except Exception as e:
        logger.warning(f"Error parsing list string: {list_str}")
        return []

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess card data with correct data types"""
    # Define dtypes for columns
    dtypes = {
        'name': str,
        'full_name': str,
        'layout': str,
        'mana_cost': str,
        'cmc': float,
        'type_line': str,
        'oracle_text': str,
        'colors': str,
        'color_identity': str,
        'power': str,
        'toughness': str,
        'rarity': str,
        'set': str,
        'collector_number': str,
        'keywords': str,
        'produced_mana': str,
        'legalities': str,
        'is_creature': bool,
        'is_land': bool,
        'is_instant_sorcery': bool,
        'is_multicolored': bool,
        'color_count': int,
        'has_etb_effect': bool,
        'is_legendary': bool
    }

    # Read CSV with specified dtypes
    df = pd.read_csv(csv_path, dtype=dtypes)
    
    # Handle any missing values
    string_columns = ['name', 'full_name', 'layout', 'mana_cost', 'type_line', 
                     'oracle_text', 'colors', 'color_identity', 'power', 'toughness',
                     'keywords', 'produced_mana']
    for col in string_columns:
        df[col] = df[col].fillna('')
    
    # Create additional columns for split card handling
    df['alternative_names'] = df['name'].apply(lambda x: [x.strip() for x in x.split('//') if x.strip()])
    df['name_lower'] = df['name'].str.lower()
    
    # Convert list-like strings to actual lists using safe parser
    list_columns = ['colors', 'color_identity', 'keywords', 'produced_mana']
    for col in list_columns:
        df[col] = df[col].apply(safe_parse_list)

    return df

class CardMechanics:
    """Analyzer for card mechanics and keywords"""
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.format_mechanics = self._identify_format_mechanics()
        
    def _identify_format_mechanics(self) -> Dict[str, int]:
        """Identify prevalent mechanics in the format"""
        mechanics = Counter()
        
        for _, card in self.card_db.iterrows():
            # Check keywords array
            if isinstance(card.keywords, list):
                mechanics.update(card.keywords)
            
            # Check oracle text for mechanics
            oracle_text = str(card.oracle_text) if pd.notna(card.oracle_text) else ''
            
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

class LandAnalyzer:
    """Enhanced land analysis system"""
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.land_cycles = self._identify_land_cycles()
        
    def _identify_land_cycles(self) -> Dict[str, List[str]]:
        """Automatically identify land cycles based on text patterns"""
        lands = self.card_db[self.card_db['is_land']].copy()
        cycles = defaultdict(list)
        
        # First handle basic lands
        for _, land in lands.iterrows():
            if 'Basic' in land['type_line']:
                cycles['basic_lands'].append(land['name'])
        
        # Remove basic lands for cycle analysis
        nonbasic_lands = lands[~lands['type_line'].str.contains('Basic', na=False)]
        
        # Common land effects to identify
        effect_patterns = {
            'damage': (r'(deals|pay|lose) \d+ (damage|life)', 'pain_lands'),
            'life_gain': (r'gain \d+ life', 'life_lands'),
            'scry': (r'scry \d+', 'scry_lands'),
            'surveil': (r'surveil \d+', 'surveil_lands'),
            'fetch': (r'search .* library .* land', 'fetch_lands'),
            'shock': (r'pay 2 life', 'shock_lands'),
            'tap_condition_two_or_fewer': (r'tapped unless .* two or fewer', 'fast_lands'),
            'tap_condition_two_or_more': (r'tapped unless .* two or more', 'slow_lands'),
            'tap_condition_basic': (r'tapped unless .* control a (Plains|Island|Swamp|Mountain|Forest)', 'check_lands'),
            'reveal_condition': (r'reveal a .* card', 'reveal_lands'),
            'dual_faced': (r'//', 'pathway_lands'),
            'triome': (r'Triome', 'triome_lands'),
            'bicycle': (r'cycling.*\{2\}', 'bicycle_lands'),
            'bounce': (r'return .* land .* hand', 'bounce_lands'),
            'storage': (r'storage counter|charge counter', 'storage_lands'),
            'man_land': (r'becomes? a.*creature.*until end of turn', 'man_lands')
        }
        
        # Identify lands that don't match any specific cycle
        for _, land in nonbasic_lands.iterrows():
            oracle_text = str(land['oracle_text']) if pd.notna(land['oracle_text']) else ''
            
            # Try to match known effects
            matched = False
            for effect, (pattern, cycle_name) in effect_patterns.items():
                if re.search(pattern, oracle_text, re.IGNORECASE):
                    cycles[cycle_name].append(land['name'])
                    matched = True
                    break
            
            # Check for specific mana production patterns
            produced_mana = land.produced_mana if isinstance(land.produced_mana, list) else []
            if produced_mana:
                if len(produced_mana) > 1:
                    if not matched:  # Only categorize if not already matched
                        cycles['dual_lands'].append(land['name'])
                elif not matched:
                    cycles['utility_lands'].append(land['name'])
            elif not matched:
                # Lands with unique effects
                cycles['special_lands'].append(land['name'])
        
        # Post-process to identify cycles by name patterns
        cycle_groups = self._group_lands_by_patterns(cycles)
        
        # Add back basic and special lands
        cycle_groups['basic_lands'] = cycles['basic_lands']
        cycle_groups['special_lands'] = cycles['special_lands']
        
        return dict(cycle_groups)
    
    def _group_lands_by_patterns(self, cycles: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Group lands by name patterns and return restructured cycles"""
        cycle_groups = defaultdict(list)
        
        # Common naming patterns for land cycles
        name_patterns = {
            'fastland': ['seachrome', 'darkslick', 'copperline', 'razorverge', 'blackcleave'],
            'slowland': ['deserted', 'deathcap', 'dreamroot', 'haunted', 'sundown'],
            'checkland': ['glacial', 'drowned', 'woodland', 'clifftop', 'isolated'],
            'shockland': ['hallowed', 'watery', 'overgrown', 'blood', 'stomping'],
            'pathway': ['brightclimb', 'clearwater', 'darkbore', 'blightstep', 'needleverge']
        }
        
        for cycle_name, lands in cycles.items():
            if cycle_name not in ['basic_lands', 'special_lands']:
                matched_lands = defaultdict(list)
                
                # Try to match lands to known naming patterns
                for land in lands:
                    land_lower = land.lower()
                    matched = False
                    
                    for pattern_name, pattern_words in name_patterns.items():
                        if any(word in land_lower for word in pattern_words):
                            matched_lands[f"{cycle_name}_{pattern_name}"].append(land)
                            matched = True
                            break
                    
                    if not matched:
                        matched_lands[cycle_name].append(land)
                
                # Add grouped lands to final cycles
                for group_name, group_lands in matched_lands.items():
                    if len(group_lands) >= 3:  # Minimum size for a cycle
                        cycle_groups[group_name].extend(group_lands)
                    else:
                        cycle_groups[cycle_name].extend(group_lands)
        
        return cycle_groups

class FormatAnalyzer:
    """Enhanced analyzer for format patterns and characteristics"""
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.mechanics_analyzer = CardMechanics(card_db)
        self.land_analyzer = LandAnalyzer(card_db)
    
    def analyze_format_characteristics(self) -> Dict[str, Any]:
        """Analyze overall format characteristics"""
        try:
            characteristics = {
                'speed_indicators': self._analyze_format_speed(),
                'mechanics': self.mechanics_analyzer.format_mechanics,
                'mana_bases': self.land_analyzer.land_cycles,
                'power_cards': self._identify_power_cards(),
                'synergy_clusters': self._identify_synergy_clusters()
            }
            return characteristics
        except Exception as e:
            logger.error(f"Error analyzing format characteristics: {str(e)}")
            return {
                'speed_indicators': {},
                'mechanics': {},
                'mana_bases': {},
                'power_cards': [],
                'synergy_clusters': {}
            }
    
    def _analyze_format_speed(self) -> Dict[str, float]:
        """Analyze format speed based on available cards"""
        try:
            nonland_cards = self.card_db[~self.card_db['is_land']]
            if nonland_cards.empty:
                return self._get_default_speed_indicators()
            
            interaction_cards = nonland_cards[
                nonland_cards['oracle_text'].str.contains('counter|destroy|exile', 
                                                        na=False, case=False)
            ]
            
            # Handle power conversion safely
            creature_cards = nonland_cards[nonland_cards['is_creature']]
            fast_threats = creature_cards[
                (creature_cards['cmc'] <= 2) & 
                creature_cards['power'].apply(
                    lambda x: str(x).isdigit() and int(str(x)) >= 2 if pd.notna(x) else False
                )
            ]
            
            return {
                'avg_cmc': nonland_cards['cmc'].mean(),
                'median_cmc': nonland_cards['cmc'].median(),
                'one_drops': len(nonland_cards[nonland_cards['cmc'] == 1]),
                'two_drops': len(nonland_cards[nonland_cards['cmc'] == 2]),
                'interaction_cmc': interaction_cards['cmc'].mean() if not interaction_cards.empty else 0,
                'creature_avg_cmc': creature_cards['cmc'].mean() if not creature_cards.empty else 0,
                'early_interaction': len(interaction_cards[interaction_cards['cmc'] <= 2]),
                'fast_threats': len(fast_threats)
            }
        except Exception as e:
            logger.error(f"Error analyzing format speed: {str(e)}")
            return self._get_default_speed_indicators()
    
    def _get_default_speed_indicators(self) -> Dict[str, float]:
        """Return default speed indicators"""
        return {
            'avg_cmc': 0.0,
            'median_cmc': 0.0,
            'one_drops': 0,
            'two_drops': 0,
            'interaction_cmc': 0.0,
            'creature_avg_cmc': 0.0,
            'early_interaction': 0,
            'fast_threats': 0
        }
    
    def _identify_power_cards(self) -> List[Dict[str, Any]]:
        """Identify potentially powerful cards based on characteristics"""
        power_cards = []
        
        for _, card in self.card_db.iterrows():
            try:
                power_score = self._calculate_power_score(card)
                if power_score >= 2:
                    power_cards.append({
                        'name': card['name'],
                        'score': power_score,
                        'cmc': card['cmc'],
                        'colors': card['colors'],
                        'type': card['type_line'],
                        'keywords': card['keywords'] if isinstance(card['keywords'], list) else []
                    })
            except Exception as e:
                logger.warning(f"Error calculating power score for {card['name']}: {str(e)}")
                continue
        
        return sorted(power_cards, key=lambda x: x['score'], reverse=True)[:20]
    
    def _calculate_power_score(self, card: pd.Series) -> float:
        """Calculate power score for a single card"""
        power_score = 0
        oracle_text = str(card['oracle_text']) if pd.notna(card['oracle_text']) else ''
        
        # Evaluate creature stats
        if card['is_creature']:
            power = str(card['power'])
            toughness = str(card['toughness'])
            if power.replace('.','',1).isdigit() and toughness.replace('.','',1).isdigit():
                total_stats = float(power) + float(toughness)
                if total_stats > 0 and card['cmc'] > 0:
                    power_score += total_stats / card['cmc']
        
        # Evaluate mana efficiency
        if card['cmc'] > 0:
            # Efficient creatures
            if card['is_creature']:
                power = str(card['power'])
                toughness = str(card['toughness'])
                if power.replace('.','',1).isdigit() and toughness.replace('.','',1).isdigit():
                    total_stats = float(power) + float(toughness)
                    if total_stats > card['cmc'] * 2:
                        power_score += 1
            
            # Efficient interaction
            if any(text in oracle_text.lower() for text in ['counter target spell', 'destroy target']):
                if card['cmc'] <= 2:
                    power_score += 1.5
                elif card['cmc'] <= 3:
                    power_score += 1
        
        # Card advantage evaluation
        card_advantage_words = ['draw', 'search your library', 'return target', 'exile']
        power_score += sum(0.5 for word in card_advantage_words if word in oracle_text.lower())
        
        # Evaluate versatility
        if card['layout'] == 'adventure' or '//' in str(card['full_name']):
            power_score += 0.5
        
        # Evaluate keywords
        valuable_keywords = ['flying', 'haste', 'ward', 'flash', 'deathtouch', 'trample', 'lifelink']
        if isinstance(card['keywords'], list):
            power_score += sum(0.3 for k in card['keywords'] if k in valuable_keywords)
        
        # Evaluate planeswalker potential
        if 'Planeswalker' in str(card['type_line']):
            power_score += 1
        
        # Evaluate board impact
        impact_phrases = ['each creature', 'all creatures', 'each player', 'each opponent']
        power_score += sum(0.5 for phrase in impact_phrases if phrase in oracle_text.lower())
        
        return power_score
    
    def _identify_synergy_clusters(self) -> Dict[str, List[str]]:
        """Identify clusters of cards with potential synergies"""
        try:
            themes = defaultdict(list)
            
            # Define synergy patterns with required types
            synergy_patterns = {
                'artifact': (r'artifact|create \w+ artifact|sacrifice an artifact', ['Artifact']),
                'token': (r'create \w+ token|whenever a creature token|creatures you control get', ['Creature']),
                'graveyard': (r'from.*graveyard|cards in graveyards|exile.*from.*graveyard', None),
                'spell_matters': (r'whenever you cast|prowess|magecraft', ['Instant', 'Sorcery']),
                'sacrifice': (r'sacrifice a creature|whenever a creature you control dies', ['Creature']),
                'counter': (r'counter target|whenever a spell is countered', ['Instant']),
                'enchantment': (r'enchantment|aura|constellation', ['Enchantment'])
            }
            
            for _, card in self.card_db.iterrows():
                try:
                    oracle_text = str(card['oracle_text']) if pd.notna(card['oracle_text']) else ''
                    type_line = str(card['type_line'])
                    
                    # Check for tribe-specific synergies
                    if 'Creature' in type_line:
                        creature_types = re.findall(r'(?<=\—\s)([^—]+?)(?=\s(?:\$|$|\}))', type_line)
                        for creature_type in creature_types:
                            if re.search(f"{creature_type}s? you control", oracle_text, re.IGNORECASE):
                                themes[f'tribal_{creature_type.lower()}'].append(card['name'])
                    
                    # Check for mechanical synergies
                    for synergy_name, (pattern, required_types) in synergy_patterns.items():
                        if re.search(pattern, oracle_text, re.IGNORECASE):
                            if required_types is None or any(t in type_line for t in required_types):
                                themes[synergy_name].append(card['name'])
                    
                    # Check for keyword-based synergies
                    if isinstance(card['keywords'], list):
                        for keyword in card['keywords']:
                            keyword_lower = keyword.lower()
                            if re.search(f"{keyword_lower}|{keyword_lower}s", oracle_text, re.IGNORECASE):
                                themes[f'{keyword_lower}_matters'].append(card['name'])
                                
                except Exception as e:
                    logger.warning(f"Error processing synergies for {card['name']}: {str(e)}")
                    continue
            
            # Filter out themes with too few or too many cards
            min_cards = 3
            max_cards = len(self.card_db) * 0.2  # 20% of total cards
            return {theme: cards for theme, cards in themes.items() 
                   if min_cards <= len(cards) <= max_cards}
                   
        except Exception as e:
            logger.error(f"Error identifying synergy clusters: {str(e)}")
            return {}

class DeckClassifier:
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.archetype_characteristics = DeckArchetype.get_characteristics()

    def _find_card_in_db(self, card_name: str) -> pd.Series:
        """Find a card in the database, handling split cards"""
        # Try exact match first
        exact_match = self.card_db[self.card_db['name'] == card_name]
        if not exact_match.empty:
            return exact_match.iloc[0]
            
        # Try case-insensitive match
        card_name_lower = card_name.lower()
        case_insensitive_match = self.card_db[self.card_db['name_lower'] == card_name_lower]
        if not case_insensitive_match.empty:
            return case_insensitive_match.iloc[0]
            
        # Try matching split card parts
        if '/' in card_name:
            parts = [part.strip().lower() for part in card_name.split('/')]
            for part in parts:
                match = self.card_db[self.card_db['name_lower'].str.contains(part, regex=False, na=False)]
                if not match.empty:
                    return match.iloc[0]
        
        raise ValueError(f"Card not found: {card_name}")
        
    def classify_deck(self, decklist: List[str]) -> Dict[str, Any]:
        """Classify deck with confidence scores for each archetype"""
        try:
            deck_stats = self._calculate_deck_statistics(decklist)
            archetype_scores = self._calculate_archetype_scores(deck_stats)
            
            if not archetype_scores:
                return {
                    'primary_archetype': DeckArchetype.UNKNOWN.value,
                    'subtype': None,
                    'confidence_scores': {},
                    'statistics': deck_stats
                }
            
            # Get primary and secondary archetypes
            sorted_scores = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
            primary_archetype = sorted_scores[0][0]
            
            # Check if deck is a hybrid
            if len(sorted_scores) > 1 and sorted_scores[1][1] > sorted_scores[0][1] * 0.8:
                classification = DeckArchetype.HYBRID.value
                subtype = f"{sorted_scores[0][0].value}-{sorted_scores[1][0].value}"
            else:
                classification = primary_archetype.value
                subtype = None
            
            return {
                'primary_archetype': classification,
                'subtype': subtype,
                'confidence_scores': archetype_scores,
                'statistics': deck_stats
            }
        except Exception as e:
            logger.error(f"Error classifying deck: {str(e)}")
            return {
                'primary_archetype': DeckArchetype.UNKNOWN.value,
                'subtype': None,
                'confidence_scores': {},
                'statistics': self._get_default_stats()
            }
    
    def _calculate_deck_statistics(self, decklist: List[str]) -> Dict[str, float]:
        """Calculate comprehensive deck statistics"""
        try:
            # Find all cards in database
            deck_cards = []
            for card_name in decklist:
                try:
                    card = self._find_card_in_db(card_name)
                    deck_cards.append(card)
                except ValueError as e:
                    logger.warning(str(e))
            
            if not deck_cards:
                return self._get_default_stats()
            
            # Separate nonland cards
            nonland_cards = [card for card in deck_cards if not card['is_land']]
            if not nonland_cards:
                return self._get_default_stats()
            
            # Calculate curve
            curve = self._calculate_curve(deck_cards)
            curve_peak = max(curve.items(), key=lambda x: x[1])[0] if curve else 0
            
            # Calculate statistics
            stats = {
                'creature_ratio': sum(1 for card in nonland_cards if card['is_creature']) / len(nonland_cards),
                'avg_cmc': np.mean([card['cmc'] for card in nonland_cards]),
                'curve_peak': curve_peak,
                'removal_ratio': self._calculate_removal_ratio(nonland_cards),
                'interaction_ratio': self._calculate_interaction_ratio(nonland_cards),
                'card_advantage_ratio': self._calculate_card_advantage_ratio(nonland_cards),
                'early_game_ratio': self._calculate_early_game_ratio(curve),
                'threat_density': self._calculate_threat_density(nonland_cards)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating deck statistics: {str(e)}")
            return self._get_default_stats()

    def _get_default_stats(self) -> Dict[str, float]:
        """Return default statistics when calculation fails"""
        return {
            'creature_ratio': 0.0,
            'avg_cmc': 0.0,
            'curve_peak': 0,
            'removal_ratio': 0.0,
            'interaction_ratio': 0.0,
            'card_advantage_ratio': 0.0,
            'early_game_ratio': 0.0,
            'threat_density': 0.0
        }
    
    def _calculate_curve(self, deck_cards: List[pd.Series]) -> Dict[int, int]:
        """Calculate mana curve of the deck"""
        curve = defaultdict(int)
        for card in deck_cards:
            if not card['is_land']:
                cmc = int(card['cmc'])
                curve[cmc] += 1
        return dict(curve)
    
    def _calculate_removal_ratio(self, nonland_cards: List[pd.Series]) -> float:
        """Calculate ratio of removal spells"""
        removal_count = 0
        removal_patterns = [
            r'destroy target', r'exile target', r'deals? \d+ damage to target',
            r'target creature gets -\d+/-\d+', r'return target.*to.*hand'
        ]
        
        for card in nonland_cards:
            oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
            if any(re.search(pattern, oracle_text) for pattern in removal_patterns):
                removal_count += 1
                
        return removal_count / len(nonland_cards) if nonland_cards else 0
    
    def _calculate_interaction_ratio(self, nonland_cards: List[pd.Series]) -> float:
        """Calculate ratio of interactive spells"""
        interaction_count = 0
        interaction_patterns = [
            r'counter target', r'can\'t attack', r'can\'t block', r'tap target',
            r'target.*doesn\'t untap', r'protection from', r'hexproof', r'ward'
        ]
        
        for card in nonland_cards:
            oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
            if any(re.search(pattern, oracle_text) for pattern in interaction_patterns):
                interaction_count += 1
                
        return interaction_count / len(nonland_cards) if nonland_cards else 0
    
    def _calculate_card_advantage_ratio(self, nonland_cards: List[pd.Series]) -> float:
        """Calculate ratio of card advantage spells"""
        advantage_count = 0
        advantage_patterns = [
            r'draw \w+ cards?', r'search your library', r'investigate',
            r'surveil \d+', r'scry \d+', r'return.*from your graveyard'
        ]
        
        for card in nonland_cards:
            oracle_text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
            if any(re.search(pattern, oracle_text) for pattern in advantage_patterns):
                advantage_count += 1
                
        return advantage_count / len(nonland_cards) if nonland_cards else 0
    
    def _calculate_early_game_ratio(self, curve: Dict[int, int]) -> float:
        """Calculate ratio of early game plays (CMC 1-2)"""
        early_game_count = sum(count for cmc, count in curve.items() if cmc <= 2)
        total_cards = sum(curve.values())
        return early_game_count / total_cards if total_cards > 0 else 0
    
    def _calculate_threat_density(self, nonland_cards: List[pd.Series]) -> float:
        """Calculate ratio of threats in the deck"""
        threat_count = 0
        
        for card in nonland_cards:
            # Consider creatures with power 3 or greater as threats
            if card['is_creature']:
                power = str(card['power'])
                if power.isdigit() and int(power) >= 3:
                    threat_count += 1
            
            # Consider planeswalkers as threats
            if 'Planeswalker' in str(card['type_line']):
                threat_count += 1
                
        return threat_count / len(nonland_cards) if nonland_cards else 0
    
    def _calculate_archetype_scores(self, deck_stats: Dict[str, float]) -> Dict[DeckArchetype, float]:
        """Calculate confidence scores for each archetype"""
        scores = {}
        
        for archetype in DeckArchetype:
            if archetype in [DeckArchetype.HYBRID, DeckArchetype.UNKNOWN]:
                continue
                
            characteristics = self.archetype_characteristics.get(archetype)
            if not characteristics:
                continue
                
            score = 0
            total_checks = 0
            
            # Check all available statistics against archetype characteristics
            if all(key in deck_stats for key in ['creature_ratio', 'removal_ratio', 'curve_peak',
                                                'interaction_ratio', 'card_advantage_ratio', 'avg_cmc']):
                # Check creature ratio
                total_checks += 1
                if characteristics.creature_ratio[0] <= deck_stats['creature_ratio'] <= characteristics.creature_ratio[1]:
                    score += 1
                
                # Check removal ratio
                total_checks += 1
                if characteristics.removal_ratio[0] <= deck_stats['removal_ratio'] <= characteristics.removal_ratio[1]:
                    score += 1
                
                # Check curve peak
                total_checks += 1
                if characteristics.curve_peak[0] <= deck_stats['curve_peak'] <= characteristics.curve_peak[1]:
                    score += 1
                
                # Check interaction ratio
                total_checks += 1
                if characteristics.interaction_ratio[0] <= deck_stats['interaction_ratio'] <= characteristics.interaction_ratio[1]:
                    score += 1
                
                # Check card advantage ratio
                total_checks += 1
                if characteristics.card_advantage_ratio[0] <= deck_stats['card_advantage_ratio'] <= characteristics.card_advantage_ratio[1]:
                    score += 1
                
                # Check average CMC
                total_checks += 1
                if characteristics.avg_cmc[0] <= deck_stats['avg_cmc'] <= characteristics.avg_cmc[1]:
                    score += 1
            
            # Calculate final score as percentage of matched characteristics
            scores[archetype] = score / total_checks if total_checks > 0 else 0
        
        return scores

class FormatMetaAnalyzer:
    """
    Integrates all components for comprehensive format analysis
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        self.format_analyzer = FormatAnalyzer(card_db)
        self.deck_classifier = DeckClassifier(card_db)
        self.format_characteristics = None
    
    def analyze_meta(self, decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Perform comprehensive meta analysis
        """
        try:
            # Analyze format characteristics first
            self.format_characteristics = self.format_analyzer.analyze_format_characteristics()
            
            # Analyze each deck
            deck_analyses = {}
            archetype_distribution = Counter()
            card_frequencies = Counter()
            
            for deck_name, decklist in decklists.items():
                try:
                    deck_analysis = self._analyze_deck(deck_name, decklist)
                    deck_analyses[deck_name] = deck_analysis
                    archetype_distribution[deck_analysis['archetype']['primary_archetype']] += 1
                    # Update frequencies only for verified cards
                    card_frequencies.update(deck_analysis['verified_cards'])
                except Exception as e:
                    logger.error(f"Error analyzing deck {deck_name}: {str(e)}")
            
            return {
                'format_characteristics': self.format_characteristics,
                'deck_analyses': deck_analyses,
                'meta_statistics': self._calculate_meta_statistics(deck_analyses),
                'archetype_distribution': dict(archetype_distribution),
                'card_frequencies': dict(card_frequencies)
            }
        except Exception as e:
            logger.error(f"Error in meta analysis: {str(e)}")
            raise
    
    def _analyze_deck(self, deck_name: str, decklist: List[str]) -> Dict[str, Any]:
        """Analyze individual deck"""
        try:
            # Verify all cards exist in database and handle split cards
            missing_cards = []
            verified_cards = []
            
            for card in decklist:
                try:
                    card_data = self.deck_classifier._find_card_in_db(card)
                    verified_cards.append(card_data['name'])
                except ValueError:
                    missing_cards.append(card)
            
            if missing_cards:
                logger.warning(f"Missing cards in {deck_name}: {missing_cards}")
            
            # Get deck classification using only verified cards
            classification = self.deck_classifier.classify_deck(verified_cards)
            
            # Analyze deck's alignment with format characteristics
            format_alignment = self._analyze_format_alignment(verified_cards)
            
            return {
                'name': deck_name,
                'archetype': classification,
                'format_alignment': format_alignment,
                'card_count': len(verified_cards),
                'missing_cards': missing_cards,
                'verified_cards': verified_cards
            }
            
        except Exception as e:
            logger.error(f"Error analyzing deck {deck_name}: {str(e)}")
            return {
                'name': deck_name,
                'archetype': {'primary_archetype': DeckArchetype.UNKNOWN.value, 
                            'statistics': self.deck_classifier._get_default_stats()},
                'format_alignment': {'mechanics_alignment': 0.0, 'power_card_alignment': 0.0},
                'card_count': len(decklist),
                'missing_cards': decklist,
                'verified_cards': []
            }
    
    def _calculate_meta_statistics(self, deck_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate comprehensive meta statistics"""
        if not deck_analyses:
            return {}
        
        # Calculate various statistics
        stats = {
            'average_statistics': self._calculate_average_stats(deck_analyses),
            'format_speed': self._evaluate_format_speed(deck_analyses),
            'color_distribution': self._analyze_color_distribution(deck_analyses),
            'archetype_matchups': self._analyze_archetype_matchups(deck_analyses),
            'meta_diversity': self._calculate_meta_diversity(deck_analyses)
        }
        
        return stats
    
    def _calculate_average_stats(self, deck_analyses: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate average statistics across all decks"""
        avg_stats = defaultdict(float)
        count = len(deck_analyses)
        
        for analysis in deck_analyses.values():
            if 'statistics' in analysis['archetype']:
                stats = analysis['archetype']['statistics']
                for key, value in stats.items():
                    avg_stats[key] += value
        
        return {key: value/count for key, value in avg_stats.items()} if count > 0 else {}
    
    def _evaluate_format_speed(self, deck_analyses: Dict[str, Dict]) -> str:
        """Evaluate overall format speed"""
        valid_analyses = [
            analysis for analysis in deck_analyses.values()
            if 'statistics' in analysis['archetype'] 
            and 'avg_cmc' in analysis['archetype']['statistics']
        ]
        
        if not valid_analyses:
            return "unknown"
            
        avg_cmc = np.mean([
            analysis['archetype']['statistics']['avg_cmc'] 
            for analysis in valid_analyses
        ])
        
        aggro_count = sum(
            1 for analysis in deck_analyses.values() 
            if analysis['archetype']['primary_archetype'] == 'aggro'
        )
        
        aggro_ratio = aggro_count / len(deck_analyses)
        
        if avg_cmc < 2.5 and aggro_ratio > 0.3:
            return "fast"
        elif avg_cmc > 3.5:
            return "slow"
        else:
            return "medium"
    
    def _analyze_color_distribution(self, deck_analyses: Dict[str, Dict]) -> Dict[str, int]:
        """Analyze color distribution in the meta"""
        color_counts = Counter()
        
        for analysis in deck_analyses.values():
            deck_colors = set()
            # Use verified_cards instead of deck name
            for card_name in analysis['verified_cards']:
                card_data = self.card_db[self.card_db['name'] == card_name]
                if not card_data.empty:
                    colors = card_data.iloc[0]['colors']
                    if isinstance(colors, list):
                        deck_colors.update(colors)
            
            if deck_colors:  # Only count if we found colors
                color_counts[tuple(sorted(deck_colors))] += 1
            
        return dict(color_counts)
    
    def _analyze_archetype_matchups(self, deck_analyses: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """Analyze theoretical archetype matchups based on characteristics"""
        archetypes = set(analysis['archetype']['primary_archetype'] 
                        for analysis in deck_analyses.values())
        
        matchups = {}
        for arch1 in archetypes:
            matchups[arch1] = {}
            for arch2 in archetypes:
                if arch1 != arch2:
                    matchups[arch1][arch2] = self._calculate_matchup_score(arch1, arch2, deck_analyses)
                else:
                    matchups[arch1][arch2] = 0.5  # Mirror match
                
        return matchups
    
    def _calculate_matchup_score(self, arch1: str, arch2: str, 
                               deck_analyses: Dict[str, Dict]) -> float:
        """Calculate theoretical matchup score between two archetypes"""
        arch1_decks = [a for a in deck_analyses.values() 
                      if a['archetype']['primary_archetype'] == arch1
                      and 'statistics' in a['archetype']]
        arch2_decks = [a for a in deck_analyses.values() 
                      if a['archetype']['primary_archetype'] == arch2
                      and 'statistics' in a['archetype']]
        
        if not arch1_decks or not arch2_decks:
            return 0.5
        
        # Compare key characteristics that influence matchups
        score = 0.5  # Start at neutral
        
        # Average stats for each archetype
        arch1_stats = self._average_archetype_stats(arch1_decks)
        arch2_stats = self._average_archetype_stats(arch2_decks)
        
        # Only adjust score if we have the necessary statistics
        if all(key in arch1_stats and key in arch2_stats 
               for key in ['avg_cmc', 'interaction_ratio', 'card_advantage_ratio']):
            if arch1_stats['avg_cmc'] < arch2_stats['avg_cmc']:
                score += 0.1
            if arch1_stats['interaction_ratio'] > arch2_stats['interaction_ratio']:
                score += 0.1
            if arch1_stats['card_advantage_ratio'] > arch2_stats['card_advantage_ratio']:
                score += 0.1
            
        return min(max(score, 0.3), 0.7)  # Keep within reasonable bounds
    
    def _average_archetype_stats(self, decks: List[Dict]) -> Dict[str, float]:
        """Calculate average statistics for an archetype"""
        avg_stats = defaultdict(float)
        count = 0
        
        for deck in decks:
            if 'statistics' in deck['archetype']:
                stats = deck['archetype']['statistics']
                for key, value in stats.items():
                    avg_stats[key] += value
                count += 1
        
        return {key: value/count for key, value in avg_stats.items()} if count > 0 else {}
    
    def _calculate_meta_diversity(self, deck_analyses: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate meta diversity metrics"""
        total_decks = len(deck_analyses)
        if total_decks == 0:
            return {
                'shannon_diversity': 0.0,
                'effective_archetypes': 0.0,
                'archetype_count': 0,
                'most_popular_archetype_share': 0.0
            }
            
        archetype_counts = Counter(
            analysis['archetype']['primary_archetype'] 
            for analysis in deck_analyses.values()
        )
        
        # Calculate Shannon diversity index
        proportions = [count/total_decks for count in archetype_counts.values()]
        shannon_diversity = -sum(p * np.log(p) for p in proportions)
        
        # Calculate effective number of archetypes
        effective_archetypes = np.exp(shannon_diversity)
        
        return {
            'shannon_diversity': shannon_diversity,
            'effective_archetypes': effective_archetypes,
            'archetype_count': len(archetype_counts),
            'most_popular_archetype_share': max(proportions)
        }
    
    def _analyze_format_alignment(self, decklist: List[str]) -> Dict[str, float]:
        """Analyze how well a deck aligns with format characteristics"""
        alignment_scores = {}
        
        # Check mechanics alignment
        deck_mechanics = self._identify_deck_mechanics(decklist)
        mechanics_overlap = set(deck_mechanics) & set(self.format_characteristics['mechanics'].keys())
        alignment_scores['mechanics_alignment'] = len(mechanics_overlap) / len(deck_mechanics) if deck_mechanics else 0
        
        # Check power card alignment
        power_cards = set(card['name'] for card in self.format_characteristics['power_cards'])
        deck_power_cards = set(decklist) & power_cards
        alignment_scores['power_card_alignment'] = len(deck_power_cards) / len(decklist) if decklist else 0
        
        return alignment_scores
    
    def _identify_deck_mechanics(self, decklist: List[str]) -> Set[str]:
        """Identify mechanics present in a deck"""
        mechanics = set()
        for card in decklist:
            card_data = self.card_db[self.card_db['name'] == card]
            if not card_data.empty:
                oracle_text = str(card_data.iloc[0]['oracle_text'])
                for mechanic in self.format_characteristics['mechanics']:
                    if mechanic.lower() in oracle_text.lower():
                        mechanics.add(mechanic)
        return mechanics

def analyze_standard_format(cards_df: pd.DataFrame, decklists: Dict[str, List[str]]) -> None:
    """Main function to analyze Standard format"""
    try:
        meta_analyzer = FormatMetaAnalyzer(cards_df)
        results = meta_analyzer.analyze_meta(decklists)
        
        # Print analysis results
        print("\n=== Standard Format Analysis ===\n")
        
        # Format speed and characteristics
        print("Format Characteristics:")
        print(f"Speed: {results['meta_statistics']['format_speed']}")
        print(f"Effective number of archetypes: {results['meta_statistics']['meta_diversity']['effective_archetypes']:.2f}")
        
        # Most common mechanics
        print("\nMost Common Mechanics:")
        for mechanic, count in sorted(results['format_characteristics']['mechanics'].items(),
                                    key=lambda x: x[1], reverse=True)[:5]:
            print(f"- {mechanic}: {count} cards")
        
        # Archetype distribution
        print("\nArchetype Distribution:")
        total_decks = sum(results['archetype_distribution'].values())
        for archetype, count in sorted(results['archetype_distribution'].items(),
                                     key=lambda x: x[1], reverse=True):
            percentage = (count / total_decks) * 100
            print(f"{archetype}: {percentage:.1f}% ({count} decks)")
        
        # Most played cards
        print("\nMost Played Cards (excluding lands):")
        for card, count in sorted(results['card_frequencies'].items(),
                                key=lambda x: x[1], reverse=True)[:10]:
            if not cards_df[cards_df['name'] == card]['is_land'].iloc[0]:
                print(f"{card}: {count} copies")
        
        # Individual deck analysis
        print("\nDeck Analysis:")
        for deck_name, analysis in results['deck_analyses'].items():
            print(f"\n{deck_name}:")
            print(f"Archetype: {analysis['archetype']['primary_archetype']}")
            if analysis['archetype']['subtype']:
                print(f"Subtype: {analysis['archetype']['subtype']}")
            print(f"Format Alignment: {analysis['format_alignment']['mechanics_alignment']:.2f}")
            if analysis['missing_cards']:
                print(f"Warning: {len(analysis['missing_cards'])} cards not found in database")
                
    except Exception as e:
        logger.error(f"Error in format analysis: {str(e)}")
        raise

def load_decklists(directory: str) -> Dict[str, List[str]]:
    """
    Load decklists from text files in the specified directory
    Returns a dictionary with deck names as keys and lists of cards as values
    """
    decklists = {}
    
    # Get all deck files
    deck_files = [f for f in os.listdir(directory) if f.startswith('Deck - ') and f.endswith('.txt')]
    
    for filename in deck_files:
        # Extract deck name from filename
        deck_name = filename.replace('Deck - ', '').replace('.txt', '')
        
        filepath = os.path.join(directory, filename)
        mainboard = []
        
        with open(filepath, 'r') as file:
            lines = file.readlines()
            
            # Process each line
            in_mainboard = True
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and use them to detect sideboard
                if not line:
                    in_mainboard = False
                    continue
                
                # Parse card entry (e.g., "4 Lightning Bolt")
                if in_mainboard:  # Only process mainboard
                    try:
                        count, card_name = line.split(' ', 1)
                        count = int(count)
                        mainboard.extend([card_name] * count)
                    except ValueError:
                        print(f"Warning: Couldn't parse line in {filename}: {line}")
        
        decklists[deck_name] = mainboard
    
    return decklists

if __name__ == "__main__":
    # Load and preprocess card database
    cards_df = load_and_preprocess_data('data/standard_cards.csv')
    
    # Load decklists
    decklists = load_decklists('current_standard_decks')
    print(f"Loaded {len(decklists)} decklists")
    
    # Run analysis
    analyze_standard_format(cards_df, decklists)