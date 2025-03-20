import re
import inflect
from typing import List, Dict, Any, Set, Tuple
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FullyDynamicDeckAnalyzer:
    """
    A fully dynamic deck name analyzer that:
    1. Extracts card types directly from the type_line structure
    2. Treats hyphenated terms as single semantic units
    3. Uses inflect for plural/singular handling
    4. Only matches against cards actually in the deck
    5. Finds cross-deck synergies by analyzing shared unidentified terms
    """
    
    def __init__(self, card_db: pd.DataFrame):
        """Initialize with the card database"""
        self.card_db = card_db
        # Cache for processed deck names
        self.name_analysis_cache = {}
        # Initialize inflect engine for handling plural/singular conversions
        self.p = inflect.engine()
        # Extract card data for faster matching (dynamically from type_line)
        self._extract_card_data_dynamically()
        # Set up the minimal hardcoded information (only color identities)
        self._setup_color_identities()
        
        print("Initialized FullyDynamicDeckAnalyzer with types extracted from card data")
    
    def _normalize_form(self, word: str) -> str:
        """Normalize a word to its singular form"""
        singular = self.p.singular_noun(word)
        return singular if singular else word
    
    def _extract_card_data_dynamically(self):
        """
        Extract card data dynamically from type_line structure
        Format is: {supertype} {type} — {subtype}
        """
        # Initialize sets for extracted data
        self.card_types = set()
        self.subtypes = set()
        self.supertypes = set()
        
        # Known supertypes in MTG for reference
        known_supertypes = {'legendary', 'basic', 'snow', 'world', 'ongoing'}
        
        # Standard main MTG card types for reference
        standard_types = {
            'artifact', 'creature', 'enchantment', 'instant', 
            'land', 'planeswalker', 'sorcery', 'battle'
        }
        
        # Process each card's type line
        for type_line in self.card_db['type_line'].dropna():
            # Handle multi-faced cards
            for face_type in type_line.split('//'):
                face_type = face_type.strip()
                
                # Skip empty lines
                if not face_type:
                    continue
                
                # Split by the em dash to separate types from subtypes
                type_parts = face_type.split('—', 1)
                main_type_section = type_parts[0].strip()
                
                # Process main type section (contains supertypes and types)
                words = main_type_section.split()
                for i, word in enumerate(words):
                    word_lower = word.lower()
                    
                    # Last word is typically the type
                    if i == len(words) - 1:
                        self.card_types.add(word_lower)
                    # Words before last are typically supertypes
                    elif word_lower in known_supertypes:
                        self.supertypes.add(word_lower)
                    # If not a known supertype but we know it's a standard type
                    elif word_lower in standard_types:
                        self.card_types.add(word_lower)
                    # Otherwise consider it a supertype
                    else:
                        self.supertypes.add(word_lower)
                
                # Process subtypes if present
                if len(type_parts) > 1:
                    subtype_section = type_parts[1].strip()
                    for subtype in subtype_section.split():
                        # Clean up any punctuation
                        clean_subtype = re.sub(r'[^\w\s]', '', subtype).strip().lower()
                        if clean_subtype:
                            self.subtypes.add(clean_subtype)
        
        # Extract keywords
        self.keywords = set()
        for kw_list in self.card_db['keywords']:
            if isinstance(kw_list, list):
                for kw in kw_list:
                    self.keywords.add(kw.lower())
            elif isinstance(kw_list, str) and kw_list.startswith('['):
                try:
                    # Handle string representation of list
                    keyword_list = self._safe_eval_list(kw_list)
                    for kw in keyword_list:
                        self.keywords.add(kw.lower())
                except Exception as e:
                    print(f"Error parsing keywords: {kw_list}. Error: {e}")
        
        # Extract oracle text words
        self.oracle_text_words = set()
        for text in self.card_db['oracle_text'].dropna():
            words = re.findall(r'\b\w+\b', text.lower())
            self.oracle_text_words.update(words)
        
        print(f"Dynamically extracted {len(self.card_types)} card types, {len(self.subtypes)} subtypes, "
              f"{len(self.supertypes)} supertypes, {len(self.keywords)} keywords")
    
    def _safe_eval_list(self, val: Any) -> List:
        """Safely evaluate string representations of lists"""
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
        elif isinstance(val, list):
            return val
            
        return []
    
    def _get_deck_card_data(self, decklist: List[str]) -> Dict[str, Dict[str, Set[str]]]:
        """
        Extract comprehensive data from the cards in a deck.
        
        Args:
            decklist: List of cards in the deck
            
        Returns:
            Dictionary mapping data types to sets of extracted information
        """
        result = {
            'name_words': set(),
            'subtypes': set(),
            'card_types': set(),
            'supertypes': set(),
            'keywords': set(),
            'oracle_words': set()
        }
        
        # Get unique cards in the deck
        unique_cards = set(decklist)
        
        # Match cards against the database
        for card_name in unique_cards:
            card_data = self._find_card_data(card_name)
            if card_data is None:
                continue
                
            # Extract words from card name
            if pd.notna(card_data['name']):
                words = re.findall(r'\b\w+\b', card_data['name'].lower())
                result['name_words'].update(words)
            
            # Extract type information
            if pd.notna(card_data['type_line']):
                self._extract_type_info(card_data['type_line'], result)
            
            # Extract keywords
            if isinstance(card_data['keywords'], list):
                for kw in card_data['keywords']:
                    result['keywords'].add(kw.lower())
            elif isinstance(card_data['keywords'], str) and card_data['keywords'].startswith('['):
                try:
                    keyword_list = self._safe_eval_list(card_data['keywords'])
                    for kw in keyword_list:
                        result['keywords'].add(kw.lower())
                except Exception:
                    pass
            
            # Extract oracle text words
            if pd.notna(card_data['oracle_text']):
                words = re.findall(r'\b\w+\b', card_data['oracle_text'].lower())
                result['oracle_words'].update(words)
        
        return result
    
    def _extract_type_info(self, type_line: str, result: Dict[str, Set[str]]):
        """
        Extract type information from a card's type line using the same logic
        as in _extract_card_data_dynamically
        
        Args:
            type_line: The type line of a card
            result: Dictionary to update with extracted info
        """
        # Handle multi-faced cards
        for face_type in type_line.split('//'):
            face_type = face_type.strip()
            
            # Skip empty lines
            if not face_type:
                continue
            
            # Split by the em dash to separate types from subtypes
            type_parts = face_type.split('—', 1)
            main_type_section = type_parts[0].strip()
            
            # Process main type section (contains supertypes and types)
            words = main_type_section.split()
            for i, word in enumerate(words):
                word_lower = word.lower()
                
                # Last word is typically the type
                if i == len(words) - 1:
                    result['card_types'].add(word_lower)
                # Words before last are typically supertypes
                elif word_lower in self.supertypes:
                    result['supertypes'].add(word_lower)
                # If not a known supertype but we know it's a standard type
                elif word_lower in self.card_types:
                    result['card_types'].add(word_lower)
                # Otherwise consider it a supertype
                else:
                    result['supertypes'].add(word_lower)
            
            # Process subtypes if present
            if len(type_parts) > 1:
                subtype_section = type_parts[1].strip()
                for subtype in subtype_section.split():
                    # Clean up any punctuation
                    clean_subtype = re.sub(r'[^\w\s]', '', subtype).strip().lower()
                    if clean_subtype:
                        result['subtypes'].add(clean_subtype)
    
    def _setup_color_identities(self):
        """Set up the minimal hardcoded information (just color names/combinations)"""
        # Color identities (guild names, shard names, etc.)
        self.color_identities = {
            'mono-white': ['W'],
            'mono-blue': ['U'],
            'mono-black': ['B'],
            'mono-red': ['R'],
            'mono-green': ['G'],
            'azorius': ['W', 'U'],
            'dimir': ['U', 'B'],
            'rakdos': ['B', 'R'],
            'gruul': ['R', 'G'],
            'selesnya': ['G', 'W'],
            'orzhov': ['W', 'B'],
            'izzet': ['U', 'R'],
            'golgari': ['B', 'G'],
            'boros': ['R', 'W'],
            'simic': ['G', 'U'],
            'esper': ['W', 'U', 'B'],
            'grixis': ['U', 'B', 'R'],
            'jund': ['B', 'R', 'G'],
            'naya': ['R', 'G', 'W'],
            'bant': ['G', 'W', 'U'],
            'abzan': ['W', 'B', 'G'],
            'jeskai': ['U', 'R', 'W'],
            'sultai': ['B', 'G', 'U'],
            'mardu': ['R', 'W', 'B'],
            'temur': ['G', 'U', 'R'],
            '4c': ['W', 'U', 'B', 'R', 'G'],
            '5c': ['W', 'U', 'B', 'R', 'G']
        }
        
        # Basic archetypes (these are universal MTG terms)
        self.basic_archetypes = {'aggro', 'midrange', 'control', 'tempo', 'combo', 'ramp'}
    
    def _extract_tokens_from_name(self, cleaned_name: str) -> Tuple[List[str], List[str]]:
        """
        Extract tokens from a deck name, preserving hyphenated terms.
        
        Args:
            cleaned_name: The cleaned deck name
            
        Returns:
            Tuple of (individual_tokens, compound_tokens)
        """
        # Extract hyphenated terms first
        hyphenated_terms = re.findall(r'\b\w+[-]\w+\b', cleaned_name.lower())
        
        # Replace hyphenated terms with placeholders to avoid splitting them
        placeholder_text = cleaned_name.lower()
        for i, term in enumerate(hyphenated_terms):
            placeholder_text = placeholder_text.replace(term, f"PLACEHOLDER{i}")
        
        # Extract individual tokens
        individual_tokens = re.findall(r'\b\w+\b', placeholder_text)
        
        # Filter out placeholder tokens
        individual_tokens = [token for token in individual_tokens if not token.startswith('PLACEHOLDER')]
        
        return individual_tokens, hyphenated_terms
    
    def process_deck_name(self, deck_name: str, decklist: List[str]) -> Dict[str, Any]:
        """
        Process a deck name to extract meaningful information.
        
        Args:
            deck_name: Name of the deck (usually a filename)
            decklist: List of cards in the deck (for precise matching)
            
        Returns:
            Dictionary with extracted information
        """
        # Check cache first
        if deck_name in self.name_analysis_cache:
            return self.name_analysis_cache[deck_name]
        
        # Clean the deck name
        if 'Deck - ' in deck_name:
            cleaned_name = deck_name.replace('Deck - ', '')
        else:
            cleaned_name = deck_name
            
        if cleaned_name.endswith('.txt'):
            cleaned_name = cleaned_name[:-4]
        
        # Initialize result
        result = {
            'original_name': deck_name,
            'cleaned_name': cleaned_name,
            'color_identity': [],
            'identified_archetypes': [],
            'identified_card_types': [],       # Card types found
            'identified_subtypes': [],         # Subtypes found
            'identified_supertypes': [],       # Supertypes found
            'identified_keywords': [],         # Keywords found
            'identified_oracle_text': [],      # Oracle text matches
            'identified_compound_terms': [],   # Compound terms like "self-bounce"
            'identified_card_references': [],  # Card name references (lowest priority)
            'unidentified_terms': []
        }
        
        # Extract data from cards in the deck
        deck_card_data = self._get_deck_card_data(decklist)
        
        # Extract individual tokens and compound tokens from the deck name
        individual_tokens, compound_tokens = self._extract_tokens_from_name(cleaned_name)
        
        # Track which tokens have been identified
        identified_tokens = set()
        identified_compounds = set()
        
        # Step 1: Identify color references
        cleaned_lower = cleaned_name.lower()
        for color_name, colors in self.color_identities.items():
            if color_name in cleaned_lower:
                result['color_identity'] = colors.copy()
                # Mark these tokens as identified
                for word in color_name.split('-'):
                    if word in individual_tokens:
                        identified_tokens.add(word)
                break
        
        # Step 2: Identify basic archetype references
        for token in individual_tokens:
            if token in self.basic_archetypes and token not in result['identified_archetypes']:
                result['identified_archetypes'].append(token)
                identified_tokens.add(token)
        
        # Step 3: Process compound tokens first (e.g., "self-bounce")
        for compound in compound_tokens:
            # Check if this compound term appears in oracle text
            compound_in_oracle = any(
                compound in card_data['oracle_text'].lower()
                for card_name in decklist
                if (card_data := self._find_card_data(card_name)) is not None
                and pd.notna(card_data['oracle_text'])
            )
            
            if compound_in_oracle:
                result['identified_compound_terms'].append({
                    'term': compound,
                    'source': 'oracle_text'
                })
                identified_compounds.add(compound)
                continue
                
            # If not found in oracle text, leave it for cross-deck analysis
            # Don't try to split these compounds - they're meaningful as a unit
        
        # Step 4: Check remaining individual tokens against card types, subtypes, etc.
        for token in individual_tokens:
            if token in identified_tokens or len(token) < 3:
                continue
            
            # Normalize token to singular form for matching
            normalized_token = self._normalize_form(token)
            
            # Check card types dynamically from the deck
            if token in deck_card_data['card_types'] or normalized_token in deck_card_data['card_types']:
                matched_token = token if token in deck_card_data['card_types'] else normalized_token
                result['identified_card_types'].append(matched_token)
                identified_tokens.add(token)
                continue
            
            # Check supertypes dynamically from the deck
            if token in deck_card_data['supertypes'] or normalized_token in deck_card_data['supertypes']:
                matched_token = token if token in deck_card_data['supertypes'] else normalized_token
                result['identified_supertypes'].append(matched_token)
                identified_tokens.add(token)
                continue
                
            # Check subtypes dynamically from the deck
            if token in deck_card_data['subtypes'] or normalized_token in deck_card_data['subtypes']:
                matched_token = token if token in deck_card_data['subtypes'] else normalized_token
                result['identified_subtypes'].append(matched_token)
                identified_tokens.add(token)
                continue
                
            # Check keywords from the deck
            if token in deck_card_data['keywords'] or normalized_token in deck_card_data['keywords']:
                matched_token = token if token in deck_card_data['keywords'] else normalized_token
                result['identified_keywords'].append(matched_token)
                identified_tokens.add(token)
                continue
            
            # Check oracle text words from the deck
            if token in deck_card_data['oracle_words'] or normalized_token in deck_card_data['oracle_words']:
                matched_token = token if token in deck_card_data['oracle_words'] else normalized_token
                result['identified_oracle_text'].append(matched_token)
                identified_tokens.add(token)
                continue
        
        # Step 5: As a last resort, check if tokens appear in card names in the deck
        for token in individual_tokens:
            if token in identified_tokens or len(token) < 3:
                continue
            
            # Check if token appears in any card name in the deck
            normalized_token = self._normalize_form(token)
            
            matching_cards = []
            
            # Look for cards in the deck that contain this token in their name
            for card_name in decklist:
                card_data = self._find_card_data(card_name)
                if card_data is None:
                    continue
                    
                name_lower = card_data['name'].lower()
                words = re.findall(r'\b\w+\b', name_lower)
                
                if token in words or normalized_token in words:
                    matching_cards.append(card_data['name'])
            
            if matching_cards:
                result['identified_card_references'].append({
                    'token': token,
                    'matching_cards': matching_cards
                })
                identified_tokens.add(token)
                continue
        
        # Step 6: Collect unidentified tokens
        for token in individual_tokens:
            if token not in identified_tokens and len(token) >= 3:
                result['unidentified_terms'].append(token)
                
        # Add unidentified compound terms
        for compound in compound_tokens:
            if compound not in identified_compounds:
                result['unidentified_terms'].append(compound)
        
        # Cache and return result
        self.name_analysis_cache[deck_name] = result
        return result
    
    def analyze_cross_deck_terms(self, 
                               decklists: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Find common oracle text patterns between decks that share unidentified terms.
        
        Args:
            decklists: Dictionary mapping deck names to card lists
            
        Returns:
            Dictionary with analysis of cross-deck terms
        """
        # First, process all deck names
        all_deck_analyses = {}
        for deck_name, decklist in decklists.items():
            all_deck_analyses[deck_name] = self.process_deck_name(deck_name, decklist)
        
        # Group decks by unidentified terms
        term_to_decks = defaultdict(list)
        for deck_name, analysis in all_deck_analyses.items():
            for term in analysis['unidentified_terms']:
                term_to_decks[term].append(deck_name)
        
        # Analyze terms that appear in multiple decks
        shared_term_analyses = {}
        for term, deck_names in term_to_decks.items():
            if len(deck_names) >= 2:  # Only analyze terms shared by at least 2 decks
                shared_term_analyses[term] = self._analyze_shared_term(term, deck_names, decklists)
        
        return shared_term_analyses
    
    def _analyze_shared_term(self, 
                           term: str, 
                           deck_names: List[str], 
                           decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze a term shared across multiple decks.
        
        Args:
            term: The shared term
            deck_names: List of deck names containing this term
            decklists: Dictionary mapping deck names to card lists
            
        Returns:
            Dictionary with analysis of the shared term
        """
        # Collect card lists from each deck
        deck_cards = {}
        for deck_name in deck_names:
            deck_cards[deck_name] = decklists.get(deck_name, [])
        
        # Find common cards across decks
        card_counts = Counter()
        for cards in deck_cards.values():
            card_counts.update(set(cards))  # Count each card once per deck
        
        common_cards = [
            (card, count) for card, count in card_counts.items()
            if count >= max(2, len(deck_names) // 2)  # In at least half of decks
        ]
        common_cards.sort(key=lambda x: x[1], reverse=True)
        
        # Collect oracle texts from common cards
        oracle_texts = []
        for card_name, _ in common_cards:
            card_data = self._find_card_data(card_name)
            if card_data is not None and pd.notna(card_data['oracle_text']):
                oracle_texts.append(card_data['oracle_text'])
        
        # Extract common terms from oracle texts
        common_terms = []
        if oracle_texts:
            try:
                vectorizer = TfidfVectorizer(
                    stop_words='english',
                    ngram_range=(1, 2),
                    max_features=20
                )
                
                # Get TF-IDF matrix
                tfidf_matrix = vectorizer.fit_transform(oracle_texts)
                
                # Sum scores across documents
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.sum(axis=0).A1
                
                # Get top terms
                top_indices = scores.argsort()[::-1][:10]
                common_terms = [(feature_names[i], float(scores[i])) for i in top_indices]
            except Exception as e:
                print(f"Error extracting terms from oracle texts: {e}")
        
        return {
            'term': term,
            'deck_count': len(deck_names),
            'deck_names': deck_names,
            'common_cards': common_cards,
            'common_terms': common_terms
        }
    
    def _find_card_data(self, card_name: str) -> pd.Series:
        """
        Find a card in the database by name.
        
        Args:
            card_name: Name of the card
            
        Returns:
            Card data as a pandas Series, or None if not found
        """
        # Try exact match
        exact_match = self.card_db[self.card_db['name'] == card_name]
        if not exact_match.empty:
            return exact_match.iloc[0]
        
        # Try case-insensitive match
        case_insensitive = self.card_db[self.card_db['name'].str.lower() == card_name.lower()]
        if not case_insensitive.empty:
            return case_insensitive.iloc[0]
        
        return None
    
    def enhance_meta_analysis(self, 
                            meta_analysis: Dict[str, Any], 
                            decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Enhance meta analysis with deck name information.
        
        Args:
            meta_analysis: Original meta analysis
            decklists: Dictionary mapping deck names to card lists
            
        Returns:
            Enhanced meta analysis
        """
        # Process all deck names
        deck_name_analyses = {}
        for deck_name, decklist in decklists.items():
            deck_name_analyses[deck_name] = self.process_deck_name(deck_name, decklist)
        
        # Analyze cross-deck terms
        cross_deck_analyses = self.analyze_cross_deck_terms(decklists)
        
        # Compile statistics, normalizing plural/singular forms
        color_identity_stats = defaultdict(int)
        archetype_stats = defaultdict(int)
        card_type_stats = defaultdict(int)
        supertype_stats = defaultdict(int)
        subtype_stats = defaultdict(int)
        keyword_stats = defaultdict(int)
        oracle_text_stats = defaultdict(int)
        compound_term_stats = defaultdict(int)
        card_reference_stats = defaultdict(int)
        
        for analysis in deck_name_analyses.values():
            # Count color identities
            color_key = ''.join(sorted(analysis['color_identity']))
            if color_key:
                color_identity_stats[color_key] += 1
            
            # Count archetypes
            for archetype in analysis['identified_archetypes']:
                archetype_stats[archetype] += 1
            
            # Count card types
            for card_type in analysis['identified_card_types']:
                card_type_stats[card_type] += 1
                
            # Count supertypes
            for supertype in analysis['identified_supertypes']:
                supertype_stats[supertype] += 1
            
            # Count subtypes (normalized to singular form)
            for subtype in analysis['identified_subtypes']:
                singular = self._normalize_form(subtype) or subtype
                subtype_stats[singular] += 1
            
            # Count keywords (normalized to singular form)
            for keyword in analysis['identified_keywords']:
                singular = self._normalize_form(keyword) or keyword
                keyword_stats[singular] += 1
            
            # Count oracle text matches
            for term in analysis['identified_oracle_text']:
                singular = self._normalize_form(term) or term
                oracle_text_stats[singular] += 1
                
            # Count compound terms
            for term_data in analysis['identified_compound_terms']:
                compound_term_stats[term_data['term']] += 1
            
            # Count card references
            for ref in analysis['identified_card_references']:
                for card in ref['matching_cards']:
                    card_reference_stats[card] += 1
        
        # Add results to meta analysis
        meta_analysis['deck_name_analysis'] = {
            'individual_analyses': deck_name_analyses,
            'cross_deck_analyses': cross_deck_analyses,
            'color_identity_stats': dict(color_identity_stats),
            'archetype_stats': dict(archetype_stats),
            'card_type_stats': dict(card_type_stats),
            'supertype_stats': dict(supertype_stats),
            'subtype_stats': dict(subtype_stats),
            'keyword_stats': dict(keyword_stats),
            'oracle_text_stats': dict(oracle_text_stats),
            'compound_term_stats': dict(compound_term_stats),
            'card_reference_stats': dict(card_reference_stats)
        }
        
        return meta_analysis
    
    def enhance_deck_analysis(self, 
                            deck_analysis: Dict[str, Any], 
                            deck_name: str, 
                            decklist: List[str]) -> Dict[str, Any]:
        """
        Enhance a single deck analysis with deck name information.
        
        Args:
            deck_analysis: Original deck analysis
            deck_name: Name of the deck
            decklist: List of cards in the deck
            
        Returns:
            Enhanced deck analysis
        """
        # Get deck name analysis
        name_analysis = self.process_deck_name(deck_name, decklist)
        
        # Add name analysis to deck analysis
        deck_analysis['name_analysis'] = name_analysis
        
        # If the analysis has strategy scores, adjust them based on identified archetypes
        if 'strategy' in deck_analysis and 'strategy_scores' in deck_analysis['strategy']:
            strategy_scores = deck_analysis['strategy']['strategy_scores'].copy()
            
            # Boost scores for identified archetypes
            for archetype in name_analysis['identified_archetypes']:
                if archetype in strategy_scores:
                    strategy_scores[archetype] += 0.15
            
            # Update strategy scores and recalculate primary strategy
            deck_analysis['strategy']['strategy_scores'] = strategy_scores
            
            # Find the new primary strategy
            primary_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            deck_analysis['strategy']['primary_strategy'] = primary_strategy
            
            # Check for hybrid archetype
            sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
            is_hybrid = (len(sorted_strategies) > 1 and 
                        sorted_strategies[1][1] > 0.8 * sorted_strategies[0][1])
            
            if is_hybrid:
                strategy_type = f"{sorted_strategies[0][0]}-{sorted_strategies[1][0]}"
            else:
                strategy_type = primary_strategy
                
            deck_analysis['strategy']['strategy_type'] = strategy_type
            deck_analysis['strategy']['is_hybrid'] = is_hybrid
        
        return deck_analysis

def enhance_mtg_semantic_analyzer(MTGSemanticAnalyzer):
    """
    Enhance the MTGSemanticAnalyzer with integrated deck name analysis.
    
    Args:
        MTGSemanticAnalyzer: The original analyzer class
        
    Returns:
        The enhanced analyzer class
    """
    # Store original methods
    original_init = MTGSemanticAnalyzer.__init__
    original_analyze_deck = MTGSemanticAnalyzer.analyze_deck
    original_analyze_meta = MTGSemanticAnalyzer.analyze_meta
    
    # Enhanced initialization
    def enhanced_init(self, card_db):
        # Call original init
        original_init(self, card_db)
        # Add deck name analyzer
        self.deck_name_analyzer = FullyDynamicDeckAnalyzer(card_db)
    
    # Enhanced deck analysis
    def enhanced_analyze_deck(self, deck_name, decklist):
        # Get original analysis
        analysis = original_analyze_deck(self, deck_name, decklist)
        # Enhance with deck name information
        return self.deck_name_analyzer.enhance_deck_analysis(analysis, deck_name, decklist)
    
    # Enhanced meta analysis
    def enhanced_analyze_meta(self, decklists):
        # Get original analysis
        meta_analysis = original_analyze_meta(self, decklists)
        # Enhance with deck name information
        return self.deck_name_analyzer.enhance_meta_analysis(meta_analysis, decklists)
    
    # Apply enhancements
    MTGSemanticAnalyzer.__init__ = enhanced_init
    MTGSemanticAnalyzer.analyze_deck = enhanced_analyze_deck
    MTGSemanticAnalyzer.analyze_meta = enhanced_analyze_meta
    
    return MTGSemanticAnalyzer

def print_enhanced_meta_report(meta_analysis):
    """
    Print enhanced meta analysis report.
    
    Args:
        meta_analysis: Enhanced meta analysis
    """
    if 'deck_name_analysis' not in meta_analysis:
        print("No deck name analysis available")
        return
    
    deck_name_analysis = meta_analysis['deck_name_analysis']
    
    print("\n===== Deck Name Analysis =====\n")
    
    # Color identity statistics
    if deck_name_analysis['color_identity_stats']:
        print("Color Identity Distribution:")
        color_names = {
            'W': 'White', 'U': 'Blue', 'B': 'Black', 'R': 'Red', 'G': 'Green'
        }
        
        for color_code, count in sorted(deck_name_analysis['color_identity_stats'].items(), 
                                      key=lambda x: x[1], reverse=True):
            # Format color identity
            colors = [color_names.get(c, c) for c in color_code]
            color_str = '/'.join(colors) if colors else "Colorless"
            
            print(f"  {color_str}: {count} decks")
    
    # Archetype statistics
    if deck_name_analysis['archetype_stats']:
        print("\nArchetype Distribution:")
        for archetype, count in sorted(deck_name_analysis['archetype_stats'].items(),
                                     key=lambda x: x[1], reverse=True):
            print(f"  {archetype.title()}: {count} decks")
    
    # Compound term statistics
    if deck_name_analysis.get('compound_term_stats'):
        print("\nCompound Terms in Deck Names:")
        for term, count in sorted(deck_name_analysis['compound_term_stats'].items(),
                                key=lambda x: x[1], reverse=True):
            print(f"  {term}: {count} occurrences")
    
    # Subtype statistics
    if deck_name_analysis['subtype_stats']:
        print("\nSubtypes in Deck Names:")
        for subtype, count in sorted(deck_name_analysis['subtype_stats'].items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {subtype.title()}: {count} occurrences")
    
    # Supertype statistics
    if deck_name_analysis.get('supertype_stats'):
        print("\nSupertypes in Deck Names:")
        for supertype, count in sorted(deck_name_analysis['supertype_stats'].items(),
                                     key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {supertype.title()}: {count} occurrences")
    
    # Keyword statistics
    if deck_name_analysis['keyword_stats']:
        print("\nKeywords in Deck Names:")
        for keyword, count in sorted(deck_name_analysis['keyword_stats'].items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {keyword.title()}: {count} occurrences")
    
    # Card reference statistics (lowest priority)
    if deck_name_analysis['card_reference_stats']:
        print("\nCard References in Deck Names:")
        for card, count in sorted(deck_name_analysis['card_reference_stats'].items(),
                                key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {card}: {count} references")
    
    # Cross-deck analyses
    if deck_name_analysis['cross_deck_analyses']:
        print("\nCross-Deck Term Analysis:")
        
        for term, analysis in deck_name_analysis['cross_deck_analyses'].items():
            print(f"\n  Term: {term} ({analysis['deck_count']} decks)")
            
            # Print deck names
            deck_names = [name.replace('Deck - ', '').replace('.txt', '') 
                          for name in analysis['deck_names']]
            print(f"  Found in: {', '.join(deck_names)}")
            
            # Print common cards
            if analysis['common_cards']:
                print("  Common Cards:")
                for card, count in analysis['common_cards'][:5]:
                    print(f"    - {card}: in {count}/{analysis['deck_count']} decks")
            
            # Print common terms from oracle texts
            if analysis['common_terms']:
                print("  Common Oracle Text Terms:")
                for term, score in analysis['common_terms'][:5]:
                    print(f"    - {term}: {score:.2f}")

# Integration point
def integrate_with_semantics_meta_analysis():
    """
    Main integration function to run the enhanced analyzer.
    """
    from semantics_meta_analysis import MTGSemanticAnalyzer, load_card_database, load_decklists
    import argparse
    import os
    import json
    
    # Parser for command-line arguments
    parser = argparse.ArgumentParser(description='Enhanced MTG Meta Analyzer')
    parser.add_argument('--cards', default='data/standard_cards.csv', help='Path to card database CSV')
    parser.add_argument('--decks', default='current_standard_decks', help='Directory containing decklists')
    parser.add_argument('--output', default='json_outputs/enhanced_semantic_meta_analysis.json', 
                       help='Output file for analysis results')
    
    args = parser.parse_args()
    
    # Load data
    try:
        print(f"Loading card database from {args.cards}...")
        card_db = load_card_database(args.cards)
        
        print(f"Loading decklists from {args.decks}...")
        decklists = load_decklists(args.decks)
        
        if not decklists:
            print("Error: No valid decklists found")
            return
            
        # Enhance the semantic analyzer
        print("Enhancing semantic analyzer with fully dynamic deck name analysis...")
        EnhancedAnalyzer = enhance_mtg_semantic_analyzer(MTGSemanticAnalyzer)
        
        # Create the enhanced analyzer
        analyzer = EnhancedAnalyzer(card_db)
        
        # Run the analysis
        print("Running enhanced meta analysis...")
        meta_analysis = analyzer.analyze_meta(decklists)
        
        # Print the reports
        from semantics_meta_analysis import print_meta_analysis_report
        print_meta_analysis_report(meta_analysis)
        print_enhanced_meta_report(meta_analysis)
        
        # Save the results
        import numpy as np
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        def json_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.float64)):
                return int(obj) if obj.is_integer() else float(obj)
            return str(obj)
        
        with open(args.output, 'w') as f:
            json.dump(meta_analysis, f, indent=2, default=json_serializer)
        
        print(f"\nEnhanced analysis results saved to {args.output}")
        
    except Exception as e:
        import traceback
        print(f"Error in analysis: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    integrate_with_semantics_meta_analysis()