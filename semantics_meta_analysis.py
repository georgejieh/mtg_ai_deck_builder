import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import re
import os
import itertools

# Import sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer

def normalize_card_name(card_name: str) -> str:
    """
    Normalize card name by standardizing single slash to double slash format.
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
    """
    Safely evaluate string representations of lists.
    """
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
        print(f"Warning: Error parsing list value: {val}. Error: {e}")
        return []

class MTGSemanticAnalyzer:
    """
    Analyzes MTG cards and decks using semantic embedding techniques
    to understand patterns without hardcoded rules.
    """
    
    def __init__(self, card_db: pd.DataFrame):
        """
        Initialize the analyzer with the card database.
        
        Args:
            card_db: DataFrame containing card information
        """
        self.card_db = card_db
        
        # Load the sentence transformer model
        # Using a small but effective model (can be replaced with larger models if accuracy is key)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Cache embeddings to avoid recomputing
        self.card_embeddings = {}
        self.deck_embeddings = {}
        
        # Process card database
        self._process_card_database()
        
        print(f"Initialized Semantic Analyzer with {len(self.card_db)} cards")
    
    def _process_card_database(self):
        """Preprocess the card database and generate embeddings"""
        # Generate embeddings for all cards
        print("Generating card embeddings...")
        
        # Prepare oracle texts
        oracle_texts = []
        valid_indices = []
        
        for i, card in self.card_db.iterrows():
            if pd.notna(card['oracle_text']):
                # Combine card name and oracle text for more context
                combined_text = f"{card['name']}. {card['oracle_text']}"
                oracle_texts.append(combined_text)
                valid_indices.append(i)
        
        # Generate embeddings in batches
        batch_size = 32
        for i in range(0, len(oracle_texts), batch_size):
            batch_texts = oracle_texts[i:i+batch_size]
            batch_indices = valid_indices[i:i+batch_size]
            
            # Generate embeddings
            batch_embeddings = self.model.encode(batch_texts)
            
            # Store embeddings
            for j, idx in enumerate(batch_indices):
                card_name = self.card_db.iloc[idx]['name']
                self.card_embeddings[card_name] = batch_embeddings[j]
        
        print(f"Generated embeddings for {len(self.card_embeddings)} cards")
    
    def analyze_deck(self, deck_name: str, decklist: List[str]) -> Dict[str, Any]:
        """
        Analyze a single deck to understand its strategy and patterns.
        
        Args:
            deck_name: Name of the deck
            decklist: List of card names in the deck
            
        Returns:
            Dictionary containing analysis results
        """
        # Match cards to database
        unique_cards = set(decklist)
        card_counts = Counter(decklist)
        
        # Match against database using the _match_cards_in_database method
        matched_cards_df = self._match_cards_in_database(unique_cards)
        valid_cards = list(matched_cards_df['name'])
        
        # Expand valid cards based on their counts in the decklist
        expanded_valid_cards = []
        for card in valid_cards:
            count = card_counts.get(card, 0)
            expanded_valid_cards.extend([card] * count)
        
        # Also check if matched cards have a full_name that matches any unmatched cards
        for _, card in matched_cards_df.iterrows():
            if pd.notna(card['full_name']) and card['full_name'] in unique_cards:
                count = card_counts.get(card['full_name'], 0)
                expanded_valid_cards.extend([card['name']] * count)
        
        if not expanded_valid_cards:
            print(f"Warning: No valid cards found in deck {deck_name}")
            return {
                'name': deck_name,
                'valid_cards': 0,
                'error': 'No valid cards found'
            }
        
        # Generate deck embedding
        deck_embedding = self._generate_deck_embedding(expanded_valid_cards)
        self.deck_embeddings[deck_name] = deck_embedding
        
        # Calculate statistical profile
        profile = self._calculate_deck_profile(expanded_valid_cards)
        
        # Find card clusters within the deck
        card_clusters = self._cluster_deck_cards(expanded_valid_cards)
        
        # Extract key themes from oracle text
        themes = self._extract_deck_themes(expanded_valid_cards)
        
        # Detect primary strategy based on card distributions
        strategy = self._detect_deck_strategy(expanded_valid_cards, profile)
        
        return {
            'name': deck_name,
            'valid_cards': len(expanded_valid_cards),
            'unique_cards': len(set(expanded_valid_cards)),
            'profile': profile,
            'card_clusters': card_clusters,
            'themes': themes,
            'strategy': strategy,
            'verified_cards': valid_cards
        }
    
    def analyze_meta(self, decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze the entire metagame to identify patterns and relationships.
        
        Args:
            decklists: Dictionary mapping deck names to card lists
            
        Returns:
            Dictionary containing meta analysis results
        """
        # Analyze each deck
        deck_analyses = {}
        for deck_name, decklist in decklists.items():
            print(f"Analyzing deck: {deck_name}")
            deck_analyses[deck_name] = self.analyze_deck(deck_name, decklist)
        
        # Cluster decks by similarity
        deck_clusters = self._cluster_decks(decklists.keys())
        
        # Find commonly played cards
        card_frequency = Counter()
        for decklist in decklists.values():
            card_frequency.update(decklist)
        
        # Find common themes across the meta
        meta_themes = self._extract_meta_themes(decklists)
        
        # Analyze format speed
        format_speed = self._analyze_format_speed(decklists)
        
        # Detect synergies dynamically without hard-coding patterns
        synergies = self._detect_meta_synergies(decklists, deck_analyses)
        
        return {
            'deck_analyses': deck_analyses,
            'deck_clusters': deck_clusters,
            'card_frequency': dict(card_frequency),
            'meta_themes': meta_themes,
            'format_speed': format_speed,
            'synergies': synergies,
            '_card_db': self.card_db  # Store for later reference
        }
    
    def _match_cards_in_database(self, card_names: set) -> pd.DataFrame:
        """
        Match card names to database entries with improved handling of split cards.
        
        Args:
            card_names: Set of card names to match
            
        Returns:
            DataFrame containing matched cards
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
        
        # For cards still not matched, try case-insensitive matching
        if remaining_cards:
            for card_name in list(remaining_cards):
                lower_name = card_name.lower()
                case_insensitive_matches = self.card_db[self.card_db['name'].str.lower() == lower_name]
                if not case_insensitive_matches.empty:
                    matched_cards = pd.concat([matched_cards, case_insensitive_matches])
                    remaining_cards.remove(card_name)
        
        # For cards still not matched, try normalized name matching (for split cards)
        if remaining_cards:
            for card_name in list(remaining_cards):
                normalized_name = normalize_card_name(card_name)
                if normalized_name != card_name:
                    # Try matching the normalized name
                    normalized_matches = self.card_db[self.card_db['name'] == normalized_name]
                    if not normalized_matches.empty:
                        matched_cards = pd.concat([matched_cards, normalized_matches])
                        remaining_cards.remove(card_name)
                    else:
                        # Also try matching normalized name against full_name
                        normalized_full_matches = self.card_db[self.card_db['full_name'] == normalized_name]
                        if not normalized_full_matches.empty:
                            matched_cards = pd.concat([matched_cards, normalized_full_matches])
                            remaining_cards.remove(card_name)
        
        # Return all matched cards
        return matched_cards
    
    def _find_card(self, card_name: str) -> Optional[pd.Series]:
        """
        Find a card in the database by name.
        
        Args:
            card_name: Name of the card to find
            
        Returns:
            Card data as a pandas Series, or None if not found
        """
        # Try exact match
        exact_match = self.card_db[self.card_db['name'] == card_name]
        if not exact_match.empty:
            return exact_match.iloc[0]
        
        # Try case-insensitive match
        case_insensitive_match = self.card_db[self.card_db['name'].str.lower() == card_name.lower()]
        if not case_insensitive_match.empty:
            return case_insensitive_match.iloc[0]
        
        # Try matching with full_name (for split cards)
        full_name_match = self.card_db[self.card_db['full_name'] == card_name]
        if not full_name_match.empty:
            return full_name_match.iloc[0]
        
        # Try with normalized name
        normalized_name = normalize_card_name(card_name)
        if normalized_name != card_name:
            normalized_match = self.card_db[self.card_db['name'] == normalized_name]
            if not normalized_match.empty:
                return normalized_match.iloc[0]
            
            # Also try normalized name against full_name
            normalized_full_match = self.card_db[self.card_db['full_name'] == normalized_name]
            if not normalized_full_match.empty:
                return normalized_full_match.iloc[0]
        
        # For split cards, try the front face
        if ' // ' in card_name or '/' in card_name:
            front_face = card_name.split(' // ')[0].split('/')[0].strip()
            front_match = self.card_db[self.card_db['name'] == front_face]
            if not front_match.empty:
                return front_match.iloc[0]
        
        return None
    
    def _generate_deck_embedding(self, decklist: List[str]) -> np.ndarray:
        """
        Generate a semantic embedding for an entire deck.
        
        Args:
            decklist: List of card names
            
        Returns:
            Numpy array containing the deck embedding
        """
        # Count card occurrences
        card_counts = Counter(decklist)
        
        # Calculate weighted average of card embeddings
        total_cards = len(decklist)
        deck_embedding = np.zeros(384)  # Embedding dimension for 'all-MiniLM-L6-v2'
        
        for card, count in card_counts.items():
            embedding = self.card_embeddings.get(card)
            if embedding is not None:
                deck_embedding += embedding * (count / total_cards)
        
        # Normalize the embedding
        norm = np.linalg.norm(deck_embedding)
        if norm > 0:
            deck_embedding = deck_embedding / norm
        
        return deck_embedding
    
    def _calculate_deck_profile(self, decklist: List[str]) -> Dict[str, Any]:
        """
        Calculate statistical profile of a deck.
        
        Args:
            decklist: List of card names
            
        Returns:
            Dictionary containing deck statistics
        """
        # Count cards by type
        type_counts = defaultdict(int)
        cmc_sum = 0
        cmc_counts = defaultdict(int)
        color_counts = defaultdict(int)
        
        for card in decklist:
            card_data = self._find_card(card)
            if card_data is not None:
                # Count card types
                if card_data['is_creature'] == True:
                    type_counts['creature'] += 1
                elif card_data['is_land'] == True:
                    type_counts['land'] += 1
                elif card_data['is_instant_sorcery'] == True:
                    type_counts['spell'] += 1
                else:
                    type_counts['other'] += 1
                
                # Count CMC
                if not card_data['is_land']:
                    cmc = int(card_data['cmc'])
                    cmc_sum += cmc
                    cmc_counts[cmc] += 1
                
                # Count colors
                colors = card_data['colors']
                if isinstance(colors, list):
                    for color in colors:
                        color_counts[color] += 1
        
        # Calculate averages
        total_cards = len(decklist)
        nonland_count = total_cards - type_counts['land']
        
        avg_cmc = cmc_sum / nonland_count if nonland_count > 0 else 0
        
        # Calculate type ratios
        type_ratios = {
            f"{type_name}_ratio": count / total_cards
            for type_name, count in type_counts.items()
        }
        
        # Determine primary colors (>20% of nonland cards)
        primary_colors = []
        for color, count in color_counts.items():
            if count / nonland_count >= 0.2:
                primary_colors.append(color)
        
        return {
            'card_count': total_cards,
            'type_counts': dict(type_counts),
            'type_ratios': type_ratios,
            'avg_cmc': avg_cmc,
            'mana_curve': dict(cmc_counts),
            'color_counts': dict(color_counts),
            'primary_colors': primary_colors
        }
    
    def _cluster_deck_cards(self, decklist: List[str]) -> List[Dict[str, Any]]:
        """
        Cluster cards within a deck to find synergy groups.
        
        Args:
            decklist: List of card names
            
        Returns:
            List of card clusters
        """
        # Get unique cards
        unique_cards = list(set(decklist))
        
        # Get embeddings for all cards
        card_embeddings = []
        valid_cards = []
        
        for card in unique_cards:
            embedding = self.card_embeddings.get(card)
            if embedding is not None:
                card_embeddings.append(embedding)
                valid_cards.append(card)
        
        if len(valid_cards) < 5:
            return []
            
        # Convert to numpy array
        embeddings_array = np.array(card_embeddings)
        
        # Apply clustering
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        clusters = clustering.fit_predict(embeddings_array)
        
        # Group cards by cluster
        cluster_groups = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            if cluster_id >= 0:  # Skip noise points (-1)
                cluster_groups[int(cluster_id)].append(valid_cards[i])
        
        # Format results
        result = []
        for cluster_id, cards in cluster_groups.items():
            # Calculate common themes in this cluster
            cluster_texts = []
            for card in cards:
                card_data = self._find_card(card)
                if card_data is not None and pd.notna(card_data['oracle_text']):
                    cluster_texts.append(card_data['oracle_text'])
            
            # Extract distinctive terms
            distinctive_terms = []
            if cluster_texts:
                try:
                    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Sum TF-IDF scores
                    tfidf_sums = tfidf_matrix.sum(axis=0).A1
                    
                    # Get top terms
                    top_indices = tfidf_sums.argsort()[-5:][::-1]
                    distinctive_terms = [feature_names[i] for i in top_indices]
                except:
                    pass
            
            result.append({
                'id': cluster_id,
                'cards': cards,
                'size': len(cards),
                'distinctive_terms': distinctive_terms
            })
        
        return result
    
    def _extract_deck_themes(self, decklist: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract key themes from the deck's oracle text.
        
        Args:
            decklist: List of card names
            
        Returns:
            Dictionary containing unigrams and bigrams
        """
        # Collect all oracle text
        oracle_texts = []
        for card in decklist:
            card_data = self._find_card(card)
            if card_data is not None and pd.notna(card_data['oracle_text']):
                oracle_texts.append(card_data['oracle_text'])
        
        if not oracle_texts:
            return {'unigrams': [], 'bigrams': []}
            
        # Extract n-grams
        try:
            # Extract unigrams and bigrams
            unigram_vectorizer = TfidfVectorizer(
                ngram_range=(1, 1),
                max_features=20,
                stop_words='english'
            )
            
            bigram_vectorizer = TfidfVectorizer(
                ngram_range=(2, 2),
                max_features=15,
                stop_words='english'
            )
            
            # Fit vectorizers
            unigram_matrix = unigram_vectorizer.fit_transform(oracle_texts)
            bigram_matrix = bigram_vectorizer.fit_transform(oracle_texts)
            
            # Get feature names
            unigram_features = unigram_vectorizer.get_feature_names_out()
            bigram_features = bigram_vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores
            unigram_sums = unigram_matrix.sum(axis=0).A1
            bigram_sums = bigram_matrix.sum(axis=0).A1
            
            # Get top terms
            top_unigram_indices = unigram_sums.argsort()[-10:][::-1]
            top_bigram_indices = bigram_sums.argsort()[-7:][::-1]
            
            top_unigrams = [
                {
                    'term': unigram_features[i],
                    'score': float(unigram_sums[i])
                }
                for i in top_unigram_indices
            ]
            
            top_bigrams = [
                {
                    'term': bigram_features[i],
                    'score': float(bigram_sums[i])
                }
                for i in top_bigram_indices
            ]
            
            return {
                'unigrams': top_unigrams,
                'bigrams': top_bigrams
            }
        except:
            return {'unigrams': [], 'bigrams': []}
    
    def _detect_deck_strategy(self, decklist: List[str], profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect the primary strategy of the deck based on card distribution.
        
        Args:
            decklist: List of card names
            profile: Statistical profile of the deck
            
        Returns:
            Dictionary containing strategy analysis
        """
        # Extract key metrics
        avg_cmc = profile['avg_cmc']
        creature_ratio = profile['type_ratios'].get('creature_ratio', 0)
        land_ratio = profile['type_ratios'].get('land_ratio', 0)
        spell_ratio = profile['type_ratios'].get('spell_ratio', 0)
        
        # Calculate aggro score
        aggro_score = 0
        if avg_cmc < 2.5:
            aggro_score += 0.4
        if creature_ratio > 0.4:
            aggro_score += 0.3
        if land_ratio < 0.4:
            aggro_score += 0.3
            
        # Calculate control score
        control_score = 0
        if avg_cmc > 3.0:
            control_score += 0.3
        if spell_ratio > 0.3:
            control_score += 0.4
        if creature_ratio < 0.3:
            control_score += 0.3
            
        # Calculate midrange score
        midrange_score = 0
        if 2.5 <= avg_cmc <= 3.5:
            midrange_score += 0.4
        if 0.3 <= creature_ratio <= 0.5:
            midrange_score += 0.3
        if spell_ratio >= 0.2 and creature_ratio >= 0.3:
            midrange_score += 0.3
            
        # Calculate combo score by analyzing card embeddings
        combo_score = self._calculate_combo_score(decklist)
        
        # Calculate tempo score
        tempo_score = 0
        if 2.0 <= avg_cmc <= 3.0:
            tempo_score += 0.3
        if creature_ratio >= 0.3 and spell_ratio >= 0.25:
            tempo_score += 0.4
        if land_ratio < 0.4:
            tempo_score += 0.3
        
        # Determine primary strategy
        strategies = {
            'aggro': aggro_score,
            'midrange': midrange_score,
            'control': control_score,
            'combo': combo_score,
            'tempo': tempo_score
        }
        
        primary_strategy = max(strategies.items(), key=lambda x: x[1])
        
        # Check for hybrid strategy
        sorted_strategies = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
        is_hybrid = sorted_strategies[1][1] > 0.8 * sorted_strategies[0][1]
        
        if is_hybrid:
            strategy_type = f"{sorted_strategies[0][0]}-{sorted_strategies[1][0]}"
        else:
            strategy_type = primary_strategy[0]
        
        return {
            'primary_strategy': primary_strategy[0],
            'strategy_type': strategy_type,
            'is_hybrid': is_hybrid,
            'strategy_scores': strategies
        }
    
    def _calculate_combo_score(self, decklist: List[str]) -> float:
        """
        Calculate the likelihood that a deck is a combo deck.
        
        Args:
            decklist: List of card names
            
        Returns:
            Float score between 0 and 1
        """
        # Calculate card similarity matrix
        card_embeddings = []
        valid_cards = []
        
        for card in set(decklist):
            embedding = self.card_embeddings.get(card)
            if embedding is not None:
                card_embeddings.append(embedding)
                valid_cards.append(card)
        
        if len(valid_cards) < 5:
            return 0.0
            
        # Convert to numpy array
        embeddings_array = np.array(card_embeddings)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings_array)
        
        # Count highly similar pairs
        high_similarity_count = np.sum(similarities > 0.7) - len(valid_cards)  # Exclude self-similarities
        
        # Normalize
        max_possible_pairs = len(valid_cards) * (len(valid_cards) - 1) / 2
        similarity_ratio = high_similarity_count / (2 * max_possible_pairs) if max_possible_pairs > 0 else 0
        
        # Check for tutors (cards that search libraries) or card draw
        special_pattern_count = 0
        for card in valid_cards:
            card_data = self._find_card(card)
            if card_data is not None and pd.notna(card_data['oracle_text']):
                oracle_text = card_data['oracle_text'].lower()
                if 'search your library' in oracle_text or 'draw' in oracle_text:
                    special_pattern_count += 1
        
        special_ratio = special_pattern_count / len(valid_cards) if len(valid_cards) > 0 else 0
        
        # Combine scores
        combo_score = 0.5 * similarity_ratio + 0.5 * min(1.0, special_ratio * 3)
        
        return combo_score
    
    def _cluster_decks(self, deck_names: List[str]) -> List[Dict[str, Any]]:
        """
        Cluster decks by similarity to find archetypes.
        
        Args:
            deck_names: List of deck names
            
        Returns:
            List of deck clusters
        """
        # Get embeddings for all decks
        valid_decks = []
        deck_embeddings_list = []
        
        for deck_name in deck_names:
            embedding = self.deck_embeddings.get(deck_name)
            if embedding is not None:
                valid_decks.append(deck_name)
                deck_embeddings_list.append(embedding)
        
        if len(valid_decks) < 2:
            return []
            
        # Convert to numpy array
        embeddings_array = np.array(deck_embeddings_list)
        
        # Apply K-means clustering
        n_clusters = min(8, len(valid_decks) // 2)  # Reasonable number of clusters
        if n_clusters < 2:
            n_clusters = 2
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings_array)
        
        # Group decks by cluster
        cluster_groups = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_groups[int(cluster_id)].append(valid_decks[i])
        
        # Calculate cluster centroids
        centroids = kmeans.cluster_centers_
        
        # Format results
        result = []
        for cluster_id, decks in cluster_groups.items():
            # Calculate closest deck to centroid
            centroid = centroids[cluster_id]
            closest_deck = None
            min_distance = float('inf')
            
            for i, deck_name in enumerate(valid_decks):
                if deck_name in decks:
                    distance = np.linalg.norm(embeddings_array[i] - centroid)
                    if distance < min_distance:
                        min_distance = distance
                        closest_deck = deck_name
            
            result.append({
                'id': cluster_id,
                'decks': decks,
                'size': len(decks),
                'centroid_deck': closest_deck
            })
        
        return result
    
    def _extract_meta_themes(self, decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Extract common themes across the metagame.
        
        Args:
            decklists: Dictionary mapping deck names to card lists
            
        Returns:
            Dictionary containing theme analysis
        """
        # Collect all oracle text
        all_oracle_text = []
        deck_oracle_texts = {}
        
        for deck_name, decklist in decklists.items():
            deck_text = []
            for card in decklist:
                card_data = self._find_card(card)
                if card_data is not None and pd.notna(card_data['oracle_text']):
                    deck_text.append(card_data['oracle_text'])
                    all_oracle_text.append(card_data['oracle_text'])
            
            if deck_text:
                deck_oracle_texts[deck_name] = ' '.join(deck_text)
        
        if not all_oracle_text:
            return {}
            
        # Extract global n-grams
        try:
            # Extract unigrams and bigrams
            unigram_vectorizer = TfidfVectorizer(
                ngram_range=(1, 1),
                max_features=30,
                stop_words='english'
            )
            
            bigram_vectorizer = TfidfVectorizer(
                ngram_range=(2, 2),
                max_features=20,
                stop_words='english'
            )
            
            # Fit vectorizers
            unigram_matrix = unigram_vectorizer.fit_transform(all_oracle_text)
            bigram_matrix = bigram_vectorizer.fit_transform(all_oracle_text)
            
            # Get feature names
            unigram_features = unigram_vectorizer.get_feature_names_out()
            bigram_features = bigram_vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores
            unigram_sums = unigram_matrix.sum(axis=0).A1
            bigram_sums = bigram_matrix.sum(axis=0).A1
            
            # Get top terms
            top_unigram_indices = unigram_sums.argsort()[-15:][::-1]
            top_bigram_indices = bigram_sums.argsort()[-10:][::-1]
            
            top_unigrams = [
                {
                    'term': unigram_features[i],
                    'score': float(unigram_sums[i])
                }
                for i in top_unigram_indices
            ]
            
            top_bigrams = [
                {
                    'term': bigram_features[i],
                    'score': float(bigram_sums[i])
                }
                for i in top_bigram_indices
            ]
            
            return {
                'unigrams': top_unigrams,
                'bigrams': top_bigrams
            }
        except:
            return {}
    
    def _analyze_format_speed(self, decklists: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze the speed of the format.
        
        Args:
            decklists: Dictionary mapping deck names to card lists
            
        Returns:
            Dictionary containing speed analysis
        """
        # Calculate average CMC across all decks
        total_cmc = 0
        total_cards = 0
        
        low_cmc_count = 0  # CMC <= 2
        mid_cmc_count = 0  # 3 <= CMC <= 4
        high_cmc_count = 0  # CMC >= 5
        
        for decklist in decklists.values():
            for card in decklist:
                card_data = self._find_card(card)
                if card_data is not None and not card_data['is_land']:
                    cmc = card_data['cmc']
                    total_cmc += cmc
                    total_cards += 1
                    
                    if cmc <= 2:
                        low_cmc_count += 1
                    elif 3 <= cmc <= 4:
                        mid_cmc_count += 1
                    else:
                        high_cmc_count += 1
        
        # Calculate averages
        avg_cmc = total_cmc / total_cards if total_cards > 0 else 0
        
        # Calculate distribution
        cmc_distribution = {
            'low_cmc_ratio': low_cmc_count / total_cards if total_cards > 0 else 0,
            'mid_cmc_ratio': mid_cmc_count / total_cards if total_cards > 0 else 0,
            'high_cmc_ratio': high_cmc_count / total_cards if total_cards > 0 else 0
        }
        
        # Determine format speed
        if avg_cmc < 2.8 and cmc_distribution['low_cmc_ratio'] > 0.5:
            speed = 'fast'
        elif avg_cmc > 3.5 or cmc_distribution['high_cmc_ratio'] > 0.2:
            speed = 'slow'
        else:
            speed = 'medium'
        
        return {
            'avg_cmc': avg_cmc,
            'cmc_distribution': cmc_distribution,
            'speed': speed
        }
    
    def _detect_meta_synergies(self, decklists: Dict[str, List[str]], 
                              deck_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect synergies across the metagame using purely data-driven methods
        without hard-coding specific mechanics.
        
        Args:
            decklists: Dictionary mapping deck names to card lists
            deck_analyses: Dictionary containing deck analyses
            
        Returns:
            Dictionary containing synergy analysis
        """
        # Identify card clusters through embedding similarity
        card_synergy_clusters = self._identify_card_synergy_clusters(decklists)
        
        # Find frequent card combinations across decks
        frequent_combinations = self._find_frequent_card_combinations(decklists)
        
        # Detect emergent themes from oracle text without predefined patterns
        emergent_themes = self._detect_emergent_themes(decklists)
        
        return {
            'card_clusters': card_synergy_clusters,
            'frequent_combinations': frequent_combinations,
            'emergent_themes': emergent_themes
        }

    def _identify_card_synergy_clusters(self, decklists: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Identify synergistic card clusters based on embedding similarity without
        relying on predefined mechanics or patterns.
    
        Args:
            decklists: Dictionary mapping deck names to card lists
        
        Returns:
            List of card cluster dictionaries
        """
        # Collect all unique cards across all decks
        all_cards = set()
        for decklist in decklists.values():
            all_cards.update(decklist)
    
        # Get embeddings for all cards that have them
        card_embeddings = {}
        for card in all_cards:
            embedding = self.card_embeddings.get(card)
            if embedding is not None:
                card_embeddings[card] = embedding
    
        # Build similarity matrix
        cards = list(card_embeddings.keys())
        if len(cards) < 5:  # Not enough cards for meaningful clustering
            return []
    
        # Convert embeddings to matrix form
        embedding_matrix = np.array([card_embeddings[card] for card in cards])
    
        try:
            # Approach 1: Use cosine_similarity and then convert to distance
            # First compute cosine similarity (ranges from -1 to 1)
            similarity_matrix = cosine_similarity(embedding_matrix)
        
            # Convert similarity to distance (0 to 2, where 0 is identical and 2 is completely dissimilar)
            # This ensures non-negative values as required by DBSCAN with precomputed metric
            distance_matrix = np.clip(1 - similarity_matrix, 0, 2)
        
            # Use DBSCAN for clustering with precomputed distances
            clustering = DBSCAN(
                eps=0.3,           # Maximum distance for points to be considered neighbors
                min_samples=3,     # Minimum points to form a core point
                metric='precomputed'  # We're providing a precomputed distance matrix
            )
        
            cluster_labels = clustering.fit_predict(distance_matrix)
        except Exception as e:
            print(f"Error in DBSCAN clustering with precomputed distances: {e}")
            print("Falling back to direct DBSCAN clustering with cosine metric")
        
            # Fallback approach: Use DBSCAN directly with cosine metric
            clustering = DBSCAN(
                eps=0.3,           # Maximum distance threshold for neighborhood
                min_samples=3,     # Minimum samples for a core point
                metric='cosine'    # Use cosine distance directly
            )
        
            cluster_labels = clustering.fit_predict(embedding_matrix)
    
        # Organize cards by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label >= 0:  # Ignore noise points (-1)
                clusters[int(label)].append(cards[i])
    
        # Convert to result format
        result = []
        for cluster_id, cluster_cards in clusters.items():
            # Only include clusters with at least 3 cards
            if len(cluster_cards) >= 3:
                # Extract common themes from cards in this cluster
                cluster_themes = self._extract_cluster_themes(cluster_cards)
            
                # Count decks containing cards from this cluster
                decks_with_cluster = []
                for deck_name, decklist in decklists.items():
                    # Count cards from this cluster in the deck
                    overlap_count = sum(1 for card in cluster_cards if card in decklist)
                
                    # If at least 2 cards or 30% of the cluster is present, count it
                    if overlap_count >= min(2, len(cluster_cards) * 0.3):
                        decks_with_cluster.append(deck_name)
            
                # Add to result
                result.append({
                    'id': cluster_id,
                    'cards': cluster_cards,
                    'size': len(cluster_cards),
                    'themes': cluster_themes,
                    'deck_count': len(decks_with_cluster),
                    'decks': decks_with_cluster
                })
    
        # Sort by number of decks (descending)
        result.sort(key=lambda x: x['deck_count'], reverse=True)
    
        return result

    def _extract_cluster_themes(self, cards: List[str]) -> List[str]:
        """
        Extract common themes from a cluster of cards using NLP techniques
        without relying on predefined patterns.
        
        Args:
            cards: List of card names
            
        Returns:
            List of theme keywords
        """
        # Collect oracle texts
        oracle_texts = []
        for card in cards:
            card_data = self._find_card(card)
            if card_data is not None and pd.notna(card_data['oracle_text']):
                oracle_texts.append(card_data['oracle_text'])
        
        if not oracle_texts:
            return []
        
        try:
            # Use TF-IDF to find distinctive terms
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 2)  # Include both unigrams and bigrams
            )
            
            tfidf_matrix = vectorizer.fit_transform(oracle_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores across all documents
            tfidf_sums = tfidf_matrix.sum(axis=0).A1
            
            # Get top terms
            top_indices = tfidf_sums.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            return top_terms
        except:
            return []

    def _find_frequent_card_combinations(self, decklists: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Find frequently co-occurring card combinations across decks
        using association rule mining techniques.
        
        Args:
            decklists: Dictionary mapping deck names to card lists
            
        Returns:
            List of frequent card combination dictionaries
        """
        # Convert decklists to transaction format
        transactions = [set(decklist) for decklist in decklists.values()]
        
        # Count all pairs of cards
        pair_counts = Counter()
        for transaction in transactions:
            # Consider all pairs of cards in the deck
            cards = list(transaction)
            for i in range(len(cards)):
                for j in range(i+1, len(cards)):
                    # Create a frozen set for the pair to use as a hashable key
                    pair = frozenset([cards[i], cards[j]])
                    pair_counts[pair] += 1
        
        # Find frequent pairs (appearing in at least 2 decks)
        frequent_pairs = {pair: count for pair, count in pair_counts.items() if count >= 2}
        
        # Extend to larger itemsets
        frequent_itemsets = []
        
        # Start with pairs
        for pair, count in frequent_pairs.items():
            frequent_itemsets.append({
                'cards': list(pair),
                'count': count,
                'support': count / len(transactions)
            })
        
        # Try to extend to triplets and beyond
        for size in range(3, 6):  # Try combinations of size 3, 4, 5
            # Count all combinations of current size
            itemset_counts = Counter()
            
            for transaction in transactions:
                # Consider all combinations of 'size' cards
                if len(transaction) >= size:
                    for combo in itertools.combinations(transaction, size):
                        itemset = frozenset(combo)
                        itemset_counts[itemset] += 1
            
            # Find frequent itemsets of current size (appearing in at least 2 decks)
            for itemset, count in itemset_counts.items():
                if count >= 2:
                    frequent_itemsets.append({
                        'cards': list(itemset),
                        'count': count,
                        'support': count / len(transactions)
                    })
        
        # Sort by count (descending)
        frequent_itemsets.sort(key=lambda x: x['count'], reverse=True)
        
        # Keep top 20 frequent itemsets
        frequent_itemsets = frequent_itemsets[:20]
        
        # Add deck information to each frequent itemset
        for itemset in frequent_itemsets:
            cards = set(itemset['cards'])
            
            # Find decks containing this itemset
            decks_with_itemset = []
            for deck_name, decklist in decklists.items():
                if cards.issubset(set(decklist)):
                    decks_with_itemset.append(deck_name)
            
            itemset['decks'] = decks_with_itemset
            itemset['deck_count'] = len(decks_with_itemset)
            
            # Extract themes
            itemset['themes'] = self._extract_cluster_themes(itemset['cards'])
        
        return frequent_itemsets

    def _detect_emergent_themes(self, decklists: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Detect emergent themes across the meta using dimensionality reduction
        and clustering on card text, without predefined mechanics.
        
        Args:
            decklists: Dictionary mapping deck names to card lists
            
        Returns:
            List of emergent theme dictionaries
        """
        # Collect all unique cards
        all_cards = set()
        for decklist in decklists.values():
            all_cards.update(decklist)
        
        # Get oracle texts
        card_texts = {}
        for card in all_cards:
            card_data = self._find_card(card)
            if card_data is not None and pd.notna(card_data['oracle_text']):
                card_texts[card] = card_data['oracle_text']
        
        if not card_texts:
            return []
        
        try:
            # Create document-term matrix using TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=500,  # Limit features to avoid sparse matrix issues
                stop_words='english'
            )
            
            # Get card names and texts in the same order
            cards = list(card_texts.keys())
            texts = [card_texts[card] for card in cards]
            
            # Create TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Apply dimensionality reduction
            from sklearn.decomposition import TruncatedSVD
            
            # Reduce to 50 dimensions (or fewer if we have fewer features)
            n_components = min(50, tfidf_matrix.shape[1] - 1, len(texts) - 1)
            if n_components < 2:  # Not enough data for meaningful reduction
                return []
                
            svd = TruncatedSVD(n_components=n_components)
            reduced_matrix = svd.fit_transform(tfidf_matrix)
            
            # Apply clustering on the reduced matrix
            from sklearn.cluster import KMeans
            
            # Determine number of clusters (between 5 and 15)
            n_clusters = min(max(5, len(texts) // 10), 15)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(reduced_matrix)
            
            # Organize cards by cluster
            theme_clusters = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                theme_clusters[int(cluster_id)].append(cards[i])
            
            # Extract themes for each cluster
            result = []
            for cluster_id, cluster_cards in theme_clusters.items():
                # Only include clusters with at least 5 cards
                if len(cluster_cards) >= 5:
                    # Extract themes
                    themes = self._extract_cluster_themes(cluster_cards)
                    
                    # Count decks using this theme
                    decks_with_theme = []
                    for deck_name, decklist in decklists.items():
                        # Count cards from this theme in the deck
                        theme_count = sum(1 for card in cluster_cards if card in decklist)
                        
                        # If at least 3 cards from this theme are present, count it
                        if theme_count >= 3:
                            decks_with_theme.append(deck_name)
                    
                    # Add to result
                    result.append({
                        'id': cluster_id,
                        'theme_name': themes[0] if themes else f"Theme {cluster_id}",
                        'cards': cluster_cards,
                        'size': len(cluster_cards),
                        'keywords': themes,
                        'deck_count': len(decks_with_theme),
                        'decks': decks_with_theme
                    })
            
            # Sort by number of decks (descending)
            result.sort(key=lambda x: x['deck_count'], reverse=True)
            
            return result
        except Exception as e:
            print(f"Error in theme detection: {e}")
            return []

# Helper functions for the main script

def print_meta_analysis_report(meta_analysis: Dict[str, Any]):
    """
    Print a concise, informative report of the meta analysis results
    with dynamically detected synergies and themes.
    
    Args:
        meta_analysis: Dictionary containing meta analysis results
    """
    print("\n===== MTG Meta Analysis Report =====\n")
    
    # Format speed
    format_speed = meta_analysis['format_speed']
    print(f"Format Speed: {format_speed['speed'].upper()}")
    print(f"Average CMC: {format_speed['avg_cmc']:.2f}")
    print(f"CMC Distribution: {format_speed['cmc_distribution']['low_cmc_ratio']*100:.0f}% Low (≤2) / " 
          f"{format_speed['cmc_distribution']['mid_cmc_ratio']*100:.0f}% Mid (3-4) / "
          f"{format_speed['cmc_distribution']['high_cmc_ratio']*100:.0f}% High (5+)")
    
    # Deck clusters
    deck_clusters = meta_analysis['deck_clusters']
    print(f"\nDeck Clusters ({len(deck_clusters)}):")
    
    # Sort clusters by size
    sorted_clusters = sorted(deck_clusters, key=lambda x: len(x['decks']), reverse=True)
    
    for i, cluster in enumerate(sorted_clusters[:5]):  # Top 5 clusters
        # Print cluster information
        print(f"  Cluster {i+1}: {len(cluster['decks'])} decks")
        print(f"  Representative Deck: {cluster['centroid_deck'].replace('Deck - ', '')}")
        
        # Show all decks if 3 or fewer, otherwise show first 3 + count
        if len(cluster['decks']) <= 3:
            print(f"  Decks: {', '.join(deck.replace('Deck - ', '') for deck in cluster['decks'])}")
        else:
            first_three = [deck.replace('Deck - ', '') for deck in cluster['decks'][:3]]
            print(f"  Decks: {', '.join(first_three)} and {len(cluster['decks'])-3} more")
        print()
    
    # Archetype distribution
    archetype_counts = Counter()
    for deck_name, analysis in meta_analysis['deck_analyses'].items():
        if 'strategy' in analysis and 'strategy_type' in analysis['strategy']:
            archetype_counts[analysis['strategy']['strategy_type']] += 1
    
    print("\nArchetype Distribution:")
    for archetype, count in archetype_counts.most_common():
        percentage = count / len(meta_analysis['deck_analyses']) * 100
        print(f"  {archetype.title()}: {count} decks ({percentage:.1f}%)")
    
    # Most played cards
    card_frequency = meta_analysis['card_frequency']
    
    # Filter out basic lands
    basic_lands = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest'}
    filtered_cards = [(card, count) for card, count in card_frequency.items() 
                     if card not in basic_lands]
    
    print("\nMost Played Cards (excluding basic lands):")
    for card, count in sorted(filtered_cards, key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {card}: {count}")
    
    # Key themes from oracle text
    meta_themes = meta_analysis['meta_themes']
    if 'unigrams' in meta_themes and meta_themes['unigrams']:
        print("\nKey Terms in Meta:")
        for term in meta_themes['unigrams'][:10]:
            print(f"  {term['term']}: {term['score']:.1f}")
    
    if 'bigrams' in meta_themes and meta_themes['bigrams']:
        print("\nKey Phrases in Meta:")
        for term in meta_themes['bigrams'][:7]:
            print(f"  {term['term']}: {term['score']:.1f}")
    
    # Synergies
    synergies = meta_analysis['synergies']
    
    # Print card clusters
    if 'card_clusters' in synergies and synergies['card_clusters']:
        print("\nCard Synergy Clusters:")
        for i, cluster in enumerate(synergies['card_clusters'][:5]):  # Top 5 clusters
            print(f"  Cluster {i+1}: {len(cluster['cards'])} cards, found in {cluster['deck_count']} decks")
            print(f"  Key Cards: {', '.join(cluster['cards'][:5])}")
            if len(cluster['cards']) > 5:
                print(f"           and {len(cluster['cards'])-5} more")
            if cluster.get('themes'):
                print(f"  Themes: {', '.join(cluster['themes'][:5])}")
            print()
    
    # Print frequent combinations
    if 'frequent_combinations' in synergies and synergies['frequent_combinations']:
        print("\nFrequent Card Combinations:")
        for i, combo in enumerate(synergies['frequent_combinations'][:5]):  # Top 5 combinations
            print(f"  Combination {i+1}: {len(combo['cards'])} cards, found in {combo['deck_count']} decks")
            print(f"  Cards: {', '.join(combo['cards'])}")
            if combo.get('themes'):
                print(f"  Themes: {', '.join(combo['themes'][:3])}")
            print()
    
    # Print emergent themes
    if 'emergent_themes' in synergies and synergies['emergent_themes']:
        print("\nEmergent Themes:")
        for i, theme in enumerate(synergies['emergent_themes'][:5]):  # Top 5 themes
            theme_name = theme.get('theme_name', f"Theme {i+1}")
            print(f"  {theme_name}: {theme['size']} cards, found in {theme['deck_count']} decks")
            print(f"  Key Cards: {', '.join(theme['cards'][:5])}")
            if len(theme['cards']) > 5:
                print(f"           and {len(theme['cards'])-5} more")
            if theme.get('keywords'):
                print(f"  Keywords: {', '.join(theme['keywords'][:5])}")
            print()
    
    # Meta insights
    print("\nMeta Insights:")
    
    # Format speed insights
    if format_speed['speed'] == 'fast':
        print("  • Fast meta with low average CMC")
        print("  • Consider lower curves and efficient interaction")
    elif format_speed['speed'] == 'slow':
        print("  • Slow meta with higher average CMC")
        print("  • Value engines and late-game threats are strong")
    else:
        print("  • Balanced meta with mixed CMC distribution")
        print("  • Flexible strategies can adapt to different matchups")
    
    # Most common archetype insight
    if archetype_counts:
        top_archetype = archetype_counts.most_common(1)[0][0]
        print(f"  • {top_archetype.title()} is the most represented archetype")

def load_card_database(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess the card database from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Process boolean columns
        bool_columns = [
            'is_creature', 'is_land', 'is_instant_sorcery',
            'is_multicolored', 'has_etb_effect', 'is_legendary'
        ]
        
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})
        
        # Process list columns (stored as strings)
        list_columns = ['colors', 'color_identity', 'keywords', 'produced_mana']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(safe_eval_list)
        
        # Convert numeric columns
        numeric_columns = ['cmc', 'color_count']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        print(f"Loaded {len(df)} cards from {csv_path}")
        return df
        
    except Exception as e:
        print(f"Error loading card database: {e}")
        raise

def load_decklists(directory: str) -> Dict[str, List[str]]:
    """
    Load decklists from text files in a directory.
    
    Args:
        directory: Path to directory containing decklist files
        
    Returns:
        Dictionary mapping deck names to card lists
    """
    decklists = {}
    
    try:
        # Find all text files
        deck_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        
        for filename in deck_files:
            filepath = os.path.join(directory, filename)
            deck_name = os.path.splitext(filename)[0]
            
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    
                    # Parse decklist
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
                        
                        # Skip sideboard cards
                        if sideboard_found:
                            continue
                        
                        # Parse card entry
                        try:
                            # Handle various formats
                            match = re.match(r'^(?:(\d+)[x]?\s+)?(.+?)(?:\s+[x]?(\d+))?$', line, re.IGNORECASE)
                            if match:
                                count = int(match.group(1) or match.group(3) or '1')
                                card_name = match.group(2).strip()
                                
                                # Add card to mainboard (respecting count)
                                mainboard.extend([card_name] * count)
                            else:
                                print(f"Warning: Could not parse line in {filename}: {line}")
                        except Exception as e:
                            print(f"Warning: Error parsing line in {filename}: {line}. Error: {e}")
                    
                    # Store decklist
                    if mainboard:
                        decklists[deck_name] = mainboard
                    
            except Exception as e:
                print(f"Error processing deck file {filename}: {e}")
        
        print(f"Loaded {len(decklists)} decklists from {directory}")
        return decklists
        
    except Exception as e:
        print(f"Error loading decklists: {e}")
        raise

def main():
    """Main execution function"""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MTG Meta Analyzer')
    parser.add_argument('--cards', default='data/standard_cards.csv', help='Path to card database CSV')
    parser.add_argument('--decks', default='current_standard_decks', help='Directory containing decklists')
    parser.add_argument('--output', default='json_outputs/semantic_meta_analysis_results.json', help='Output file for analysis results')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        # Load data
        card_db = load_card_database(args.cards)
        decklists = load_decklists(args.decks)
        
        if not decklists:
            print("Error: No valid decklists found")
            return
        
        # Debug mode - compare card names
        if args.debug:
            print("\n=== Debug: Card Name Comparison ===")
            # Get all unique card names from decklists
            all_decklist_cards = set()
            for decklist in decklists.values():
                all_decklist_cards.update(decklist)
            
            # Get all card names from database
            all_db_cards = set(card_db['name'])
            
            # Check which cards from decklists are missing in database
            missing_cards = all_decklist_cards - all_db_cards
            if missing_cards:
                print(f"Found {len(missing_cards)} cards in decklists that don't exactly match database names:")
                for i, card in enumerate(sorted(missing_cards)):
                    if i < 20:  # Show first 20 only
                        print(f"  - '{card}'")
                    elif i == 20:
                        print(f"  - ... and {len(missing_cards) - 20} more")
                        break
            
            # Look for similar names
            print("\nChecking for similar names:")
            for deck_card in list(missing_cards)[:10]:  # Check first 10 missing cards
                deck_card_lower = deck_card.lower()
                matches = []
                
                for db_card in all_db_cards:
                    # Check for partial string containment
                    if (deck_card_lower in db_card.lower() or 
                        db_card.lower() in deck_card_lower):
                        similarity = max(
                            len(deck_card_lower) / len(db_card.lower()) if len(db_card) > 0 else 0,
                            len(db_card.lower()) / len(deck_card_lower) if len(deck_card) > 0 else 0
                        )
                        if similarity > 0.7:  # Only show reasonably similar names
                            matches.append((db_card, similarity))
                
                # Show top matches
                if matches:
                    top_matches = sorted(matches, key=lambda x: x[1], reverse=True)[:3]
                    print(f"  '{deck_card}' might match:")
                    for match, similarity in top_matches:
                        print(f"    - '{match}' (similarity: {similarity:.2f})")
                else:
                    print(f"  '{deck_card}' has no close matches in database")
        
        # Initialize analyzer
        analyzer = MTGSemanticAnalyzer(card_db)
        
        # Perform meta analysis
        print("\nAnalyzing meta...")
        meta_analysis = analyzer.analyze_meta(decklists)
        
        # Print report
        print_meta_analysis_report(meta_analysis)
        
        # Save results
        import json
        
        def json_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.float64)):
                return int(obj) if obj.is_integer() else float(obj)
            return str(obj)
        
        with open(args.output, 'w') as f:
            json.dump(meta_analysis, f, indent=2, default=json_serializer)
        
        print(f"\nAnalysis results saved to {args.output}")
        
    except Exception as e:
        print(f"Error in meta analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()