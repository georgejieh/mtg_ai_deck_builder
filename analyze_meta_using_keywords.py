import os
import re
import json
import logging
import sys
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
    Dynamically extracts all types, supertypes, and subtypes from the card database.
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        
        # Extract all types, supertypes, and subtypes
        self.all_types, self.all_supertypes, self.all_subtypes = self._extract_all_type_information()
        
        # Extract all keywords
        self.all_keywords = self._extract_all_keywords()
        
        logger.info(f"Meta Card Analyzer initialized with:")
        logger.info(f"- {len(self.all_types)} card types")
        logger.info(f"- {len(self.all_supertypes)} supertypes")
        logger.info(f"- {len(self.all_subtypes)} subtypes")
        logger.info(f"- {len(self.all_keywords)} keywords")
    
    def _extract_all_type_information(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Extract all card types, supertypes, and subtypes from the card database.
        
        Returns:
            Tuple containing sets of all types, supertypes, and subtypes
        """
        # Standard main MTG card types (for reference)
        standard_types = {
            'artifact', 'creature', 'enchantment', 'instant', 
            'land', 'planeswalker', 'sorcery', 'battle'
        }
        
        # Known supertypes in MTG (for reference)
        known_supertypes = {
            'basic', 'legendary', 'ongoing', 'snow', 'world'
        }
        
        # Initialize sets
        all_types = set()
        all_supertypes = set()
        all_subtypes = set()
        
        # Process each card type line
        for type_line in self.card_db['type_line'].dropna():
            # Handle multiple faces
            for face_type in type_line.split('//'):
                face_type = face_type.strip()
                
                # Skip empty lines
                if not face_type:
                    continue
                
                # Split by the em dash to separate types from subtypes
                type_subtype_parts = face_type.split('—', 1)
                
                # Process the left side (types and supertypes)
                type_parts = type_subtype_parts[0].strip().split()
                
                # Last word before the dash is usually the card type
                for word in type_parts:
                    word = word.lower()
                    
                    # Skip empty strings
                    if not word:
                        continue
                    
                    # Add to appropriate set
                    if word.lower() in standard_types:
                        all_types.add(word.lower())
                    elif word.lower() in known_supertypes:
                        all_supertypes.add(word.lower())
                    else:
                        # If we're not sure, check its position
                        if word == type_parts[-1]:
                            # Last word is likely a type
                            all_types.add(word.lower())
                        else:
                            # Words before the last are likely supertypes
                            all_supertypes.add(word.lower())
                
                # Process subtypes if present
                if len(type_subtype_parts) > 1:
                    subtype_part = type_subtype_parts[1].strip()
                    for subtype in subtype_part.split():
                        # Clean up any punctuation
                        subtype = re.sub(r'[^\w\s]', '', subtype).strip().lower()
                        if subtype:
                            all_subtypes.add(subtype)
        
        # Add any missing standard types (in case they weren't found)
        all_types.update(standard_types)
        all_supertypes.update(known_supertypes)
        
        return all_types, all_supertypes, all_subtypes
    
    def _extract_all_keywords(self) -> Set[str]:
        """
        Extract all unique keywords from the card database.
        
        Returns:
            Set of all unique keywords
        """
        all_keywords = set()
        
        for keywords in self.card_db['keywords']:
            if isinstance(keywords, list):
                for keyword in keywords:
                    all_keywords.add(keyword.lower())
            elif isinstance(keywords, str) and keywords.startswith('['):
                # Handle string representation of list
                try:
                    keyword_list = safe_eval_list(keywords)
                    for keyword in keyword_list:
                        all_keywords.add(keyword.lower())
                except Exception as e:
                    logger.warning(f"Error parsing keywords: {keywords}. Error: {e}")
        
        return all_keywords
    
    def analyze_card(self, card: pd.Series) -> Dict[str, Set[str]]:
        """
        Analyze a single card and extract all its characteristics.
        
        Args:
            card: DataFrame row representing a card
            
        Returns:
            Dictionary with sets of types, supertypes, subtypes, and keywords
        """
        result = {
            'types': set(),
            'supertypes': set(),
            'subtypes': set(),
            'keywords': set(),
            'references': set()
        }
        
        # Process type line if present
        if pd.notna(card['type_line']):
            # Process each face of the card
            for face_type in card['type_line'].split('//'):
                face_type = face_type.strip()
                
                # Skip empty faces
                if not face_type:
                    continue
                
                # Split by dash to separate types from subtypes
                type_parts = face_type.split('—', 1)
                type_section = type_parts[0].strip().lower()
                
                # Extract types and supertypes
                for word in type_section.split():
                    if word in self.all_types:
                        result['types'].add(word)
                    elif word in self.all_supertypes:
                        result['supertypes'].add(word)
                
                # Extract subtypes if present
                if len(type_parts) > 1:
                    subtype_section = type_parts[1].strip()
                    for subtype in subtype_section.split():
                        # Clean up any punctuation
                        clean_subtype = re.sub(r'[^\w\s]', '', subtype).strip().lower()
                        if clean_subtype and clean_subtype in self.all_subtypes:
                            result['subtypes'].add(clean_subtype)
        
        # Process keywords if present
        if isinstance(card['keywords'], list):
            for keyword in card['keywords']:
                result['keywords'].add(keyword.lower())
        elif isinstance(card['keywords'], str) and card['keywords'].startswith('['):
            # Handle string representation of list
            try:
                keyword_list = safe_eval_list(card['keywords'])
                for keyword in keyword_list:
                    result['keywords'].add(keyword.lower())
            except Exception as e:
                logger.warning(f"Error parsing keywords: {card['keywords']}. Error: {e}")
        
        # Extract references from oracle text
        if pd.notna(card['oracle_text']):
            oracle_text = card['oracle_text'].lower()
            
            # Check for references to types
            for type_name in self.all_types:
                if re.search(r'\b' + re.escape(type_name) + r'\b', oracle_text):
                    result['references'].add(f"{type_name}_reference")
            
            # Check for references to subtypes
            for subtype in self.all_subtypes:
                if re.search(r'\b' + re.escape(subtype) + r'\b', oracle_text):
                    result['references'].add(f"{subtype}_reference")
        
        return result
    
    def analyze_meta_cards(self, meta_cards_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """
        Analyze all cards in the meta and count various card characteristics.
        
        Args:
            meta_cards_df: DataFrame containing cards in the meta
            
        Returns:
            Dictionary with counts for types, supertypes, subtypes, keywords, and references
        """
        # Initialize result dictionaries
        result = {
            'types': defaultdict(int),
            'supertypes': defaultdict(int),
            'subtypes': defaultdict(int),
            'keywords': defaultdict(int),
            'references': defaultdict(int)
        }
        
        # Create a copy to avoid SettingWithCopyWarning
        meta_cards = meta_cards_df.copy()
        
        # Process each card
        for _, card in meta_cards.iterrows():
            card_analysis = self.analyze_card(card)
            
            # Update counts for each category
            for category in ['types', 'supertypes', 'subtypes', 'keywords', 'references']:
                for item in card_analysis[category]:
                    result[category][item] += 1
        
        # Convert defaultdicts to regular dicts for cleaner output
        return {
            category: dict(counts) for category, counts in result.items()
        }

class DynamicArchetypeClassifier:
    """
    Advanced deck archetype classifier that focuses on statistical patterns
    rather than specific mechanic detection.
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        # Limit to core archetypes that can be reliably detected
        self.archetypes = ['aggro', 'midrange', 'control', 'ramp', 'tempo']
        # Define archetype characteristics dynamically
        self.archetype_characteristics = self._define_archetype_characteristics()
        logger.info("Dynamic Archetype Classifier initialized")
    
    def _define_archetype_characteristics(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Define core characteristics of each archetype based on statistical patterns
        rather than specific card functions.
        """
        return {
            'aggro': {
                'creature_ratio': (0.4, 0.8),
                'avg_cmc': (1.0, 2.5),
                'low_cmc_creature_ratio': (0.3, 1.0),
                'high_cmc_creature_ratio': (0.0, 0.2)
            },
            'midrange': {
                'creature_ratio': (0.3, 0.6),
                'avg_cmc': (2.5, 3.5),
                'low_cmc_creature_ratio': (0.1, 0.4),
                'high_cmc_creature_ratio': (0.1, 0.3)
            },
            'control': {
                'creature_ratio': (0.0, 0.4),
                'avg_cmc': (2.5, 4.5),
                'instant_sorcery_ratio': (0.3, 0.7),
                'low_cmc_creature_ratio': (0.0, 0.2)
            },
            'tempo': {
                'creature_ratio': (0.3, 0.5),
                'avg_cmc': (1.5, 3.0),
                'low_cmc_creature_ratio': (0.2, 0.5),
                'instant_sorcery_ratio': (0.2, 0.5)
            },
            'ramp': {
                'creature_ratio': (0.2, 0.5),
                'avg_cmc': (2.5, 4.5),
                'high_cmc_creature_ratio': (0.2, 0.5),
                'land_ratio': (0.35, 0.5)
            }
        }
    
    def classify_deck(self, decklist: List[str]) -> Dict[str, float]:
        """
        Classify a deck into archetypes with confidence scores.
        
        Args:
            decklist: List of card names in the deck
            
        Returns:
            Dictionary mapping archetype names to confidence scores (0-1)
        """
        # Match cards in the decklist to the database
        deck_cards = self._match_cards_to_database(decklist)
        
        if deck_cards.empty:
            logger.warning("No cards from the decklist were found in the database")
            return {archetype: 0.0 for archetype in self.archetypes}
        
        # Calculate deck statistics
        deck_stats = self._calculate_deck_statistics(deck_cards, decklist)
        
        # Calculate archetype scores
        archetype_scores = self._calculate_archetype_scores(deck_stats)
        
        # Detect hybrid archetypes
        hybrid_scores = self._detect_hybrid_archetypes(archetype_scores)
        
        # Combine and normalize scores
        combined_scores = {**archetype_scores, **hybrid_scores}
        normalized_scores = self._normalize_scores(combined_scores)
        
        return normalized_scores
    
    def _match_cards_to_database(self, decklist: List[str]) -> pd.DataFrame:
        """
        Match cards in the decklist to entries in the database.
        
        Args:
            decklist: List of card names
            
        Returns:
            DataFrame containing matched cards
        """
        # Normalize card names
        normalized_names = [normalize_card_name(name) for name in decklist]
        
        # Create a set of unique card names for faster lookup
        unique_cards = set(normalized_names)
        
        # Match against database
        matched_cards = self.card_db[self.card_db['name'].isin(unique_cards)]
        
        # If no cards were matched, try matching with case-insensitive comparison
        if matched_cards.empty:
            unique_cards_lower = {name.lower() for name in unique_cards}
            matched_cards = self.card_db[self.card_db['name'].str.lower().isin(unique_cards_lower)]
        
        return matched_cards
    
    def _calculate_deck_statistics(self, deck_cards: pd.DataFrame, decklist: List[str]) -> Dict[str, float]:
        """
        Calculate various statistics for a deck, focusing on card distribution rather
        than specific mechanics.
        
        Args:
            deck_cards: DataFrame containing matched cards
            decklist: Original decklist (to count copies)
            
        Returns:
            Dictionary of deck statistics
        """
        # Count the number of cards matched from the decklist
        card_counts = Counter(decklist)
        total_cards = sum(card_counts.values())
        
        # Count lands
        lands = deck_cards[deck_cards['is_land'] == True]
        land_count = sum(card_counts[card] for card in lands['name'] if card in card_counts)
        
        # Non-land cards
        nonland_cards = deck_cards[deck_cards['is_land'] != True]
        nonland_count = sum(card_counts[card] for card in nonland_cards['name'] if card in card_counts)
        
        # Creature cards
        creature_cards = deck_cards[deck_cards['is_creature'] == True]
        creature_count = sum(card_counts[card] for card in creature_cards['name'] if card in card_counts)
        
        # Instant/sorcery cards
        instant_sorcery_cards = deck_cards[deck_cards['is_instant_sorcery'] == True]
        instant_sorcery_count = sum(card_counts[card] for card in instant_sorcery_cards['name'] if card in card_counts)
        
        # Low CMC creatures (CMC <= 2)
        low_cmc_creatures = creature_cards[creature_cards['cmc'] <= 2]
        low_cmc_creature_count = sum(card_counts[card] for card in low_cmc_creatures['name'] if card in card_counts)
        
        # High CMC creatures (CMC >= 4)
        high_cmc_creatures = creature_cards[creature_cards['cmc'] >= 4]
        high_cmc_creature_count = sum(card_counts[card] for card in high_cmc_creatures['name'] if card in card_counts)
        
        # Calculate ratios
        stats = {
            'land_ratio': land_count / total_cards if total_cards > 0 else 0,
            'creature_ratio': creature_count / nonland_count if nonland_count > 0 else 0,
            'instant_sorcery_ratio': instant_sorcery_count / nonland_count if nonland_count > 0 else 0,
            'low_cmc_creature_ratio': low_cmc_creature_count / creature_count if creature_count > 0 else 0,
            'high_cmc_creature_ratio': high_cmc_creature_count / creature_count if creature_count > 0 else 0,
            'avg_cmc': nonland_cards['cmc'].mean() if not nonland_cards.empty else 0,
            'median_cmc': nonland_cards['cmc'].median() if not nonland_cards.empty else 0
        }
        
        return stats
    
    def _calculate_archetype_scores(self, deck_stats: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate how well a deck matches each archetype.
        
        Args:
            deck_stats: Dictionary of deck statistics
            
        Returns:
            Dictionary mapping archetype names to confidence scores
        """
        archetype_scores = {}
        
        for archetype, characteristics in self.archetype_characteristics.items():
            score = 0
            total_factors = 0
            
            for characteristic, (min_val, max_val) in characteristics.items():
                if characteristic in deck_stats:
                    value = deck_stats[characteristic]
                    
                    # Calculate how well this characteristic matches the archetype
                    if min_val <= value <= max_val:
                        # Perfect match
                        factor_score = 1.0
                    elif value < min_val:
                        # Below range, partial score based on how close to min
                        factor_score = max(0, value / min_val)
                    else:  # value > max_val
                        # Above range, partial score based on how close to max
                        factor_score = max(0, 1 - (value - max_val) / max_val)
                    
                    score += factor_score
                    total_factors += 1
            
            # Calculate final score
            archetype_scores[archetype] = score / total_factors if total_factors > 0 else 0
        
        return archetype_scores
    
    def _detect_hybrid_archetypes(self, archetype_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Detect potential hybrid archetypes by combining scores.
        
        Args:
            archetype_scores: Dictionary of base archetype scores
            
        Returns:
            Dictionary of hybrid archetype scores
        """
        hybrid_scores = {}
        
        # Sort archetypes by score
        sorted_archetypes = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Look for potential combinations of top-scoring archetypes
        for i in range(len(sorted_archetypes) - 1):
            archetype1, score1 = sorted_archetypes[i]
            
            for j in range(i + 1, len(sorted_archetypes)):
                archetype2, score2 = sorted_archetypes[j]
                
                # Only consider high-scoring archetypes
                if score1 >= 0.3 and score2 >= 0.3:
                    # The hybrid score is the average of the two, with a boost for synergy
                    hybrid_name = f"{archetype1}-{archetype2}"
                    hybrid_score = (score1 + score2) / 2 * 1.1  # Small boost for hybrid
                    
                    # Make sure the score doesn't exceed 1
                    hybrid_scores[hybrid_name] = min(1.0, hybrid_score)
        
        return hybrid_scores
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to sum to 1.
        
        Args:
            scores: Dictionary of scores
            
        Returns:
            Dictionary of normalized scores
        """
        total = sum(scores.values())
        
        if total > 0:
            return {k: v / total for k, v in scores.items()}
        
        return scores

class MetaAnalyzer:
    """
    Main meta analysis system that processes decklists and identifies card usage patterns.
    Now includes deck archetype classification.
    """
    def __init__(self, card_db: pd.DataFrame):
        self.card_db = card_db
        # Create indices for faster lookups
        self._create_indices()
        # Initialize card analyzer
        self.card_analyzer = MetaCardAnalyzer(card_db)
        # Initialize archetype classifier
        self.archetype_classifier = DynamicArchetypeClassifier(card_db)
        logger.info("Meta Analyzer initialized with Card Analyzer and Archetype Classifier")
    
    def _create_indices(self):
        """Create indices for efficient card lookups"""
        # Name-based indices
        self.name_to_card = dict(zip(self.card_db['name'], self.card_db.to_dict('records')))
        self.name_lower_to_card = dict(zip(self.card_db['name'].str.lower(), self.card_db.to_dict('records')))
        
        # Create basic land set for filtering
        self.basic_lands = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest'}
    
    def analyze_meta(self, decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """Perform meta analysis on a collection of decklists"""
        try:
            # Convert decklists to card objects for efficient processing
            processed_decklists = self._process_decklists(decklists)
            
            # Calculate meta speed
            meta_speed = self._calculate_meta_speed(processed_decklists)
            logger.info(f"Meta speed calculated as {meta_speed['speed']}")
            
            # Analyze format characteristics
            format_characteristics = self._analyze_format_characteristics(processed_decklists)
            logger.info("Format characteristics analyzed")
            
            # Collect card frequencies
            card_frequencies = Counter()
            for deck_info in processed_decklists.values():
                for card, count in deck_info['card_counts'].items():
                    if card not in self.basic_lands:
                        card_frequencies[card] += count
            
            # Classify deck archetypes
            archetype_analysis = self._analyze_deck_archetypes(decklists)
            logger.info("Deck archetypes analyzed")
            
            # Calculate meta statistics
            meta_statistics = self._calculate_meta_statistics(meta_speed, card_frequencies, len(processed_decklists))
            
            logger.info("Meta analysis completed successfully")
            return {
                'meta_speed': meta_speed,
                'format_characteristics': format_characteristics,
                'meta_statistics': meta_statistics,
                'card_frequencies': dict(card_frequencies),
                'archetype_analysis': archetype_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in meta analysis: {str(e)}")
            raise
    
    def _analyze_deck_archetypes(self, decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze the archetypes of all decks in the meta.
        
        Args:
            decklists: Dictionary mapping deck names to card lists
            
        Returns:
            Dictionary containing archetype analysis results
        """
        deck_archetypes = {}
        archetype_counts = Counter()
        
        # Classify each deck
        for deck_name, card_list in decklists.items():
            try:
                # Get archetype scores for this deck
                archetype_scores = self.archetype_classifier.classify_deck(card_list)
                
                # Store the result
                deck_archetypes[deck_name] = archetype_scores
                
                # Determine the primary archetype for this deck
                if archetype_scores:
                    primary_archetype = max(archetype_scores.items(), key=lambda x: x[1])[0]
                    archetype_counts[primary_archetype] += 1
            except Exception as e:
                logger.warning(f"Error classifying deck {deck_name}: {e}")
                continue
        
        # Calculate the meta breakdown
        total_decks = len(decklists)
        meta_breakdown = {
            archetype: {
                'count': count,
                'percentage': (count / total_decks) * 100 if total_decks > 0 else 0
            }
            for archetype, count in archetype_counts.items()
        }
        
        return {
            'deck_archetypes': deck_archetypes,
            'meta_breakdown': meta_breakdown
        }
    
    def _process_decklists(self, decklists: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Process decklists into a more efficient format with improved handling of split cards"""
        processed = {}
        for deck_name, card_list in decklists.items():
            # Normalize card names in the decklist (convert single slash to double slash)
            normalized_card_list = [normalize_card_name(card) for card in card_list]
        
            # Count cards
            card_counts = Counter(normalized_card_list)
        
            # Match cards in database
            matched_cards_df = self._match_cards_in_database(card_counts.keys())
        
            # Better detection of missing cards by checking both name and full_name
            found_names = set(matched_cards_df['name'])
            found_full_names = set()
        
            # Extract valid full names from the matched cards
            for _, card in matched_cards_df.iterrows():
                if pd.notna(card['full_name']):
                    found_full_names.add(card['full_name'])
        
            # A card is only missing if it's not in found_names AND not in found_full_names
            missing_cards = set()
            for card_name in card_counts.keys():
                if card_name not in found_names and card_name not in found_full_names:
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
        """
        Match card names to database entries with improved handling of split cards
        """
        # Create an empty DataFrame to store matched cards
        matched_cards = pd.DataFrame()
        remaining_cards = set(card_names)
        
        # Try exact name match
        exact_matches = self.card_db[self.card_db['name'].isin(remaining_cards)]
        if not exact_matches.empty:
            matched_cards = pd.concat([matched_cards, exact_matches])
            remaining_cards -= set(exact_matches['name'])
        
        # For remaining cards, check if they match anything in the full_name column
        if remaining_cards:
            full_name_matches = self.card_db[self.card_db['full_name'].isin(remaining_cards)]
            if not full_name_matches.empty:
                matched_cards = pd.concat([matched_cards, full_name_matches])
                remaining_cards -= set(full_name_matches['full_name'])
        
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
                
                # If count is 0, try checking if this card's full_name is in the card_counts
                if count == 0 and pd.notna(card['full_name']):
                    count = deck_info['card_counts'].get(card['full_name'], 0)
                
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
                'early_game_ratio': 0.0
            }
        
        avg_cmc = total_cmc / total_cards
        early_game_ratio = early_game_count / total_cards
        
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
            'early_game_ratio': early_game_ratio
        }
    
    def _analyze_format_characteristics(self, processed_decklists: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
        """Analyze format characteristics using our dynamic approach"""
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
        
def print_meta_analysis_report(meta_analysis: Dict[str, Any]):
    """Print meta analysis report with enhanced archetype information"""
    print("\n=== Magic Format Meta Analysis Report ===\n")
    
    # Meta Speed
    speed_info = meta_analysis['meta_speed']
    print("1. Meta Speed:")
    print(f"   Speed: {speed_info['speed'].title()}")
    print(f"   Average CMC: {speed_info['avg_cmc']:.2f}")
    print(f"   Early Game Ratio: {speed_info['early_game_ratio']*100:.1f}%")
    
    # Archetype Breakdown
    if 'archetype_analysis' in meta_analysis:
        print("\n2. Meta Archetype Breakdown:")
        sorted_archetypes = sorted(
            meta_analysis['archetype_analysis']['meta_breakdown'].items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )
        
        for archetype, data in sorted_archetypes:
            print(f"   {archetype.title()}: {data['count']} decks ({data['percentage']:.1f}%)")
    
    # Card Types
    print("\n3. Card Types in Meta:")
    for type_name, count in sorted(meta_analysis['format_characteristics']['types'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {type_name.title()}: {count}")
    
    # Supertypes
    print("\n4. Supertypes in Meta:")
    for supertype, count in sorted(meta_analysis['format_characteristics']['supertypes'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {supertype.title()}: {count}")
    
    # Keywords
    print("\n5. Keywords in Meta:")
    for keyword, count in sorted(meta_analysis['format_characteristics']['keywords'].items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"   {keyword.title()}: {count}")
    
    # Subtypes
    print("\n6. Most Common Subtypes:")
    for subtype, count in sorted(meta_analysis['format_characteristics']['subtypes'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {subtype.title()}: {count}")
    
    # References
    print("\n7. Type References in Oracle Text:")
    for ref, count in sorted(meta_analysis['format_characteristics']['references'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {ref.replace('_reference', '').replace('_', ' ').title()}: {count}")
    
    # Most Played Cards
    print("\n8. Most Played Cards:")
    for card_info in meta_analysis['meta_statistics']['most_played_cards']:
        print(f"   {card_info['card']}: {card_info['count']} copies")
    
    # Key Cards
    print("\n9. Key Cards (found in multiple decks):")
    key_cards = meta_analysis['meta_statistics']['key_cards']
    for i, card in enumerate(key_cards[:20]):
        print(f"   {card}")
        if i > 0 and i % 10 == 0 and i < len(key_cards) - 1:
            print("   ...")
            break
    
    # Individual Deck Archetype Analysis (if requested)
    if 'archetype_analysis' in meta_analysis and '--verbose' in sys.argv:
        print("\n10. Individual Deck Archetype Analysis:")
        for deck_name, archetype_scores in meta_analysis['archetype_analysis']['deck_archetypes'].items():
            # Get the primary archetype
            primary_archetype = max(archetype_scores.items(), key=lambda x: x[1])[0]
            primary_score = archetype_scores[primary_archetype]
            
            print(f"   {deck_name}: {primary_archetype.title()} ({primary_score:.2f})")
            
            # Show secondary archetypes if they have significant scores
            secondary_archetypes = [(arch, score) for arch, score in archetype_scores.items() 
                                   if arch != primary_archetype and score > 0.25]
            
            if secondary_archetypes:
                secondary_str = ", ".join([f"{arch.title()} ({score:.2f})" 
                                          for arch, score in sorted(secondary_archetypes, 
                                                                   key=lambda x: x[1], 
                                                                   reverse=True)])
                print(f"     Secondary: {secondary_str}")

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
            default='meta_keyword_analysis_results.json',
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
