import json
import statistics
from collections import Counter, defaultdict

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: File {file_path} is not valid JSON.")
        return {}

def reconcile_meta_speed(json_files):
    """Combine meta speed information from different sources."""
    speeds = []
    avg_cmc_values = []
    early_game_ratios = []
    interaction_ratios = []
    
    # Extract meta speed information from each JSON
    if 'parse_meta' in json_files and 'meta_speed' in json_files['parse_meta']:
        speeds.append(json_files['parse_meta']['meta_speed']['speed'])
        avg_cmc_values.append(json_files['parse_meta']['meta_speed']['avg_cmc'])
        early_game_ratios.append(json_files['parse_meta']['meta_speed']['early_game_ratio'])
        if 'interaction_ratio' in json_files['parse_meta']['meta_speed']:
            interaction_ratios.append(json_files['parse_meta']['meta_speed']['interaction_ratio'])
    
    if 'semantic_meta' in json_files and 'format_speed' in json_files['semantic_meta']:
        speeds.append(json_files['semantic_meta']['format_speed']['speed'])
        avg_cmc_values.append(json_files['semantic_meta']['format_speed']['avg_cmc'])
        early_game_ratios.append(json_files['semantic_meta']['format_speed']['cmc_distribution']['low_cmc_ratio'])
    
    if 'enhanced_meta' in json_files and 'format_speed' in json_files['enhanced_meta']:
        speeds.append(json_files['enhanced_meta']['format_speed']['speed'])
        avg_cmc_values.append(json_files['enhanced_meta']['format_speed']['avg_cmc'])
        early_game_ratios.append(json_files['enhanced_meta']['format_speed']['cmc_distribution']['low_cmc_ratio'])
    
    if 'keyword_meta' in json_files and 'meta_speed' in json_files['keyword_meta']:
        speeds.append(json_files['keyword_meta']['meta_speed']['speed'])
        avg_cmc_values.append(json_files['keyword_meta']['meta_speed']['avg_cmc'])
        early_game_ratios.append(json_files['keyword_meta']['meta_speed']['early_game_ratio'])
    
    # Calculate consolidated values
    most_common_speed = Counter(speeds).most_common(1)[0][0] if speeds else "unknown"
    avg_cmc = statistics.mean(avg_cmc_values) if avg_cmc_values else 0
    early_game_ratio = statistics.mean(early_game_ratios) if early_game_ratios else 0
    
    result = {
        "speed": most_common_speed,
        "avg_cmc": avg_cmc,
        "early_game_ratio": early_game_ratio,
    }
    
    if interaction_ratios:
        result["interaction_ratio"] = statistics.mean(interaction_ratios)
    
    # Add speed characteristics 
    if most_common_speed == "fast":
        result["speed_characteristics"] = {
            "description": "Fast metas favor low-curve decks with efficient threats",
            "recommended_cmc_range": [1.6, 2.8],
            "recommended_curve_peak": [1, 2],
            "recommended_land_count": [20, 24]
        }
    elif most_common_speed == "medium":
        result["speed_characteristics"] = {
            "description": "Medium-speed metas favor balanced decks with a mix of threats and answers",
            "recommended_cmc_range": [2.5, 3.5],
            "recommended_curve_peak": [2, 3],
            "recommended_land_count": [23, 26]
        }
    elif most_common_speed == "slow":
        result["speed_characteristics"] = {
            "description": "Slow metas favor control decks with powerful late-game plays",
            "recommended_cmc_range": [3.0, 4.0],
            "recommended_curve_peak": [3, 4],
            "recommended_land_count": [25, 28]
        }
    
    return result

def extract_mana_curve_patterns(json_files):
    """Extract mana curve patterns across different archetypes."""
    curve_patterns = {}
    
    # Try to extract from semantic_meta or enhanced_meta deck profiles
    for source in ['semantic_meta', 'enhanced_meta']:
        if source in json_files and 'deck_analyses' in json_files[source]:
            archetype_curves = defaultdict(list)
            
            for deck_name, analysis in json_files[source]['deck_analyses'].items():
                if ('strategy' in analysis and 'profile' in analysis and 
                    'mana_curve' in analysis['profile']):
                    
                    archetype = analysis['strategy']['primary_strategy']
                    if analysis['strategy'].get('is_hybrid') and 'strategy_type' in analysis['strategy']:
                        archetype = analysis['strategy']['strategy_type']
                    
                    # Normalize and store curve
                    curve = analysis['profile']['mana_curve']
                    normalized_curve = {}
                    total_cards = sum(int(count) for count in curve.values())
                    
                    for cmc, count in curve.items():
                        normalized_curve[cmc] = count / total_cards if total_cards > 0 else 0
                    
                    archetype_curves[archetype].append(normalized_curve)
            
            # Calculate average curve for each archetype
            for archetype, curves in archetype_curves.items():
                if len(curves) >= 2:  # Only include if we have at least 2 samples
                    avg_curve = defaultdict(float)
                    for curve in curves:
                        for cmc, ratio in curve.items():
                            avg_curve[cmc] += ratio / len(curves)
                    
                    # Sort by CMC
                    sorted_curve = {str(k): v for k, v in sorted(avg_curve.items(), key=lambda x: float(x[0]))}
                    
                    # Find curve peak
                    peak_cmc = max(avg_curve.items(), key=lambda x: x[1])[0] if avg_curve else "unknown"
                    
                    curve_patterns[archetype] = {
                        "average_distribution": sorted_curve,
                        "peak_cmc": peak_cmc,
                        "sample_size": len(curves)
                    }
            
            # If we found curve patterns, no need to check the other source
            if curve_patterns:
                break
    
    return curve_patterns

def analyze_card_type_distributions(json_files):
    """Analyze card type distributions across different archetypes."""
    type_distributions = {}
    
    # Try to extract from semantic_meta or enhanced_meta deck profiles
    for source in ['semantic_meta', 'enhanced_meta']:
        if source in json_files and 'deck_analyses' in json_files[source]:
            archetype_types = defaultdict(lambda: defaultdict(list))
            archetype_ratios = defaultdict(lambda: defaultdict(list))
            
            for deck_name, analysis in json_files[source]['deck_analyses'].items():
                if ('strategy' in analysis and 'profile' in analysis and 
                    'type_counts' in analysis['profile']):
                    
                    archetype = analysis['strategy']['primary_strategy']
                    if analysis['strategy'].get('is_hybrid') and 'strategy_type' in analysis['strategy']:
                        archetype = analysis['strategy']['strategy_type']
                    
                    # Store type counts
                    type_counts = analysis['profile']['type_counts']
                    total_cards = sum(int(count) for count in type_counts.values())
                    
                    # Store both raw counts and ratios
                    for type_name, count in type_counts.items():
                        archetype_types[archetype][type_name].append(count)
                        # Calculate and store ratio (percentage of deck)
                        if total_cards > 0:
                            ratio = count / total_cards
                            archetype_ratios[archetype][type_name].append(ratio)
            
            # Calculate average counts and ratios for each archetype
            for archetype, type_data in archetype_types.items():
                if sum(len(counts) for counts in type_data.values()) >= 5:  # Minimum sample size
                    avg_types = {}
                    avg_ratios = {}
                    
                    for type_name, counts in type_data.items():
                        if counts:
                            avg_types[type_name] = statistics.mean(counts)
                    
                    for type_name, ratios in archetype_ratios[archetype].items():
                        if ratios:
                            avg_ratios[type_name] = statistics.mean(ratios)
                    
                    type_distributions[archetype] = {
                        "average_counts": avg_types,
                        "average_ratios": avg_ratios,  # Include ratios which are more useful
                        "sample_size": len(list(type_data.values())[0]) if type_data else 0
                    }
            
            # If we found type distributions, no need to check the other source
            if type_distributions:
                break
    
    return type_distributions

def extract_color_combinations(json_files):
    """Extract popular color combinations."""
    color_combos = Counter()
    color_details = defaultdict(lambda: {"decks": [], "archetypes": Counter()})
    
    # Extract from semantic_meta or enhanced_meta deck profiles
    for source in ['semantic_meta', 'enhanced_meta']:
        if source in json_files and 'deck_analyses' in json_files[source]:
            for deck_name, analysis in json_files[source]['deck_analyses'].items():
                if 'profile' in analysis and 'primary_colors' in analysis['profile']:
                    combo = ''.join(sorted(analysis['profile']['primary_colors']))
                    color_combos[combo] += 1
                    color_details[combo]["decks"].append(deck_name)
                    
                    if 'strategy' in analysis:
                        archetype = analysis['strategy']['primary_strategy']
                        if analysis['strategy'].get('is_hybrid') and 'strategy_type' in analysis['strategy']:
                            archetype = analysis['strategy']['strategy_type']
                        color_details[combo]["archetypes"][archetype] += 1
    
    # Fallback to parse_meta color counts
    if len(color_combos) == 0 and 'parse_meta' in json_files and 'deck_analyses' in json_files['parse_meta']:
        for deck_name, analysis in json_files['parse_meta']['deck_analyses'].items():
            if 'colors' in analysis:
                colors = [color for color, count in analysis['colors'].items() if count > 0]
                combo = ''.join(sorted(colors))
                color_combos[combo] += 1
                color_details[combo]["decks"].append(deck_name)
    
    # Enhanced by deck_name_analysis if available
    if 'enhanced_meta' in json_files and 'deck_name_analysis' in json_files['enhanced_meta']:
        name_analysis = json_files['enhanced_meta']['deck_name_analysis']
        if 'color_identity_stats' in name_analysis:
            for color_code, count in name_analysis['color_identity_stats'].items():
                color_combos[color_code] += count
    
    # Convert to percentage and sort by popularity
    total_decks = sum(color_combos.values())
    color_data = []
    
    for combo, count in color_combos.most_common():
        # Get dominant archetype for this color combo
        archetypes = color_details[combo]["archetypes"]
        dominant_archetype = archetypes.most_common(1)[0][0] if archetypes else "unknown"
        
        # Calculate archetype diversity (higher = more diverse)
        archetype_diversity = len(archetypes) / count if count > 0 else 0
        
        color_data.append({
            "combination": combo,
            "count": count,
            "percentage": (count / total_decks) * 100 if total_decks > 0 else 0,
            "dominant_archetype": dominant_archetype,
            "archetype_diversity": archetype_diversity
        })
    
    return color_data

def find_defining_cards(json_files):
    """Identify defining cards for each archetype."""
    archetype_cards = defaultdict(Counter)
    deck_count_by_archetype = Counter()
    all_decks_count = 0
    
    # First, gather cards by archetype from semantic or enhanced meta
    for source in ['semantic_meta', 'enhanced_meta']:
        if source in json_files and 'deck_analyses' in json_files[source]:
            for deck_name, analysis in json_files[source]['deck_analyses'].items():
                if 'strategy' in analysis and 'verified_cards' in analysis:
                    all_decks_count += 1
                    archetype = analysis['strategy']['primary_strategy']
                    if analysis['strategy'].get('is_hybrid') and 'strategy_type' in analysis['strategy']:
                        archetype = analysis['strategy']['strategy_type']
                    
                    deck_count_by_archetype[archetype] += 1
                    for card in analysis['verified_cards']:
                        archetype_cards[archetype][card] += 1
    
    # Now find defining cards for each archetype
    defining_cards = {}
    for archetype, cards in archetype_cards.items():
        # Only consider archetypes with enough samples
        if deck_count_by_archetype[archetype] < 2:
            continue
        
        # Calculate the relevance score for each card in this archetype
        # (how much more often it appears in this archetype vs. others)
        card_scores = {}
        for card, count in cards.items():
            # Percentage of decks in this archetype that play this card
            archetype_percentage = count / deck_count_by_archetype[archetype]
            
            # Percentage of all decks that play this card
            appearances_in_other_archetypes = sum(
                other_cards[card] for arch, other_cards in archetype_cards.items() 
                if arch != archetype
            )
            total_percentage = (count + appearances_in_other_archetypes) / all_decks_count if all_decks_count > 0 else 0
            
            # Relevance = how much more likely this card appears in this archetype
            # than in general (with a minimum threshold to avoid division by zero)
            relevance = archetype_percentage / max(0.01, total_percentage)
            
            # Only include cards that appear in a significant portion of the archetype's decks
            if archetype_percentage >= 0.5:
                card_scores[card] = {
                    "count": count,
                    "percentage": archetype_percentage * 100,
                    "relevance": relevance
                }
        
        # Sort by relevance and include top cards
        sorted_cards = sorted(
            card_scores.items(), 
            key=lambda x: (x[1]["relevance"], x[1]["percentage"]), 
            reverse=True
        )
        
        defining_cards[archetype] = [
            {"name": card, **score} for card, score in sorted_cards[:10]
        ]
    
    return defining_cards

def extract_cluster_patterns(json_files):
    """Extract patterns from deck clusters to identify meta trends."""
    cluster_patterns = []
    
    # Look for cluster information in semantic_meta or enhanced_meta
    for source in ['semantic_meta', 'enhanced_meta']:
        if source in json_files and 'deck_clusters' in json_files[source]:
            clusters = json_files[source]['deck_clusters']
            total_decks_in_sample = len(json_files[source]['deck_analyses']) if 'deck_analyses' in json_files[source] else 0
            
            for cluster in clusters:
                if cluster['size'] >= 3:  # Only consider significant clusters
                    # Get the centroid deck
                    centroid_name = cluster.get('centroid_deck', '')
                    centroid_strategy = None
                    
                    if centroid_name and 'deck_analyses' in json_files[source]:
                        centroid_analysis = json_files[source]['deck_analyses'].get(centroid_name, {})
                        if 'strategy' in centroid_analysis:
                            centroid_strategy = centroid_analysis['strategy'].get('primary_strategy')
                    
                    cluster_patterns.append({
                        "id": cluster['id'],
                        "size": cluster['size'],
                        "decks": cluster['decks'],
                        "centroid_deck": centroid_name,
                        "dominant_strategy": centroid_strategy,
                        "sample_proportion": (cluster['size'] / total_decks_in_sample) * 100 if total_decks_in_sample > 0 else 0,
                        "note": "This proportion represents prevalence in our sample only, not actual meta share"
                    })
            
            # If we found cluster patterns, no need to check the other source
            if cluster_patterns:
                break
    
    # Sort by cluster size
    cluster_patterns.sort(key=lambda x: x['size'], reverse=True)
    
    return cluster_patterns

def analyze_card_clusters(json_files):
    """Analyze card clusters to identify synergy groups."""
    card_clusters = []
    
    # Try sources in order of preference
    for source in ['enhanced_meta', 'semantic_meta']:
        if source in json_files and 'synergies' in json_files[source]:
            synergies = json_files[source]['synergies']
            
            # Extract from card_clusters
            if 'card_clusters' in synergies:
                for cluster in synergies['card_clusters']:
                    if cluster['size'] >= 2:  # Minimum cluster size
                        card_clusters.append({
                            "cards": cluster['cards'],
                            "themes": cluster.get('themes', [])[:5],  # Top 5 themes
                            "frequency": cluster.get('deck_count', 0),
                            "source": "semantic"
                        })
            
            # Add from frequent_combinations
            if 'frequent_combinations' in synergies:
                for combo in synergies['frequent_combinations']:
                    if combo.get('count', 0) >= 3 and len(combo['cards']) >= 2:  # Minimum frequency and size
                        card_clusters.append({
                            "cards": combo['cards'],
                            "themes": combo.get('themes', [])[:5],  # Top 5 themes
                            "frequency": combo.get('count', 0),
                            "support": combo.get('support', 0),
                            "source": "frequent_combinations"
                        })
            
            # If we found card clusters, no need to check the other source
            if card_clusters:
                break
    
    # Sort by frequency
    card_clusters.sort(key=lambda x: x.get('frequency', 0), reverse=True)
    
    return card_clusters

def extract_card_type_trends(json_files):
    """Extract trends in card types and subtypes."""
    type_trends = {}
    
    # Get information from keyword_meta or parse_meta
    for source in ['keyword_meta', 'parse_meta']:
        if source in json_files and 'format_characteristics' in json_files[source]:
            format_chars = json_files[source]['format_characteristics']
            
            # Main types
            if 'types' in format_chars:
                types = format_chars['types']
                total_types = sum(types.values())
                
                type_trends['main_types'] = [
                    {"type": type_name, "count": count, "percentage": (count / total_types) * 100 if total_types > 0 else 0}
                    for type_name, count in sorted(types.items(), key=lambda x: x[1], reverse=True)
                ]
            
            # Subtypes
            if 'subtypes' in format_chars:
                subtypes = format_chars['subtypes']
                total_subtypes = sum(subtypes.values())
                
                type_trends['subtypes'] = [
                    {"subtype": subtype_name, "count": count, "percentage": (count / total_subtypes) * 100 if total_subtypes > 0 else 0}
                    for subtype_name, count in sorted(subtypes.items(), key=lambda x: x[1], reverse=True)
                    if count >= 3  # Filter out rare subtypes
                ]
            
            # References to types in card text
            if 'references' in format_chars:
                references = format_chars['references']
                total_references = sum(references.values())
                
                type_trends['type_references'] = [
                    {"reference": ref_name, "count": count, "percentage": (count / total_references) * 100 if total_references > 0 else 0}
                    for ref_name, count in sorted(references.items(), key=lambda x: x[1], reverse=True)
                    if count >= 3  # Filter out rare references
                ]
            
            # If we found type trends, no need to check the other source
            if type_trends:
                break
    
    return type_trends

def analyze_archetype_interactions(json_files):
    """Analyze how different archetypes interact with each other in the meta."""
    archetype_tiers = {"prevalent": [], "common": [], "uncommon": []}
    archetype_characteristics = {}
    
    # Check if actual meta representation data is available
    if 'meta_representation' in json_files:
        for archetype_data in json_files['meta_representation']:
            meta_percentage = archetype_data.get('meta_percentage', 0)
            name = archetype_data.get('archetype', 'unknown')
            deck_count = archetype_data.get('deck_count', 0)
            
            archetype_info = {
                "name": name,
                "meta_percentage": meta_percentage,
                "deck_count": deck_count
            }
            
            # Categorize based on meta percentage
            if meta_percentage >= 5.0:
                archetype_tiers["prevalent"].append(archetype_info)
            elif meta_percentage >= 1.0:
                archetype_tiers["common"].append(archetype_info)
            else:
                archetype_tiers["uncommon"].append(archetype_info)
    
    # If no meta_representation, use sample data from semantic or enhanced meta
    elif 'enhanced_meta' in json_files or 'semantic_meta' in json_files:
        source = 'enhanced_meta' if 'enhanced_meta' in json_files else 'semantic_meta'
        if 'deck_analyses' in json_files[source]:
            archetype_counter = Counter()
            
            for deck_name, analysis in json_files[source]['deck_analyses'].items():
                if 'strategy' in analysis:
                    archetype = analysis['strategy']['primary_strategy']
                    if analysis['strategy'].get('is_hybrid') and 'strategy_type' in analysis['strategy']:
                        archetype = analysis['strategy']['strategy_type']
                    
                    archetype_counter[archetype] += 1
            
            total_sample_decks = sum(archetype_counter.values())
            
            # Categorize based on sample prevalence
            for archetype, count in archetype_counter.items():
                sample_percentage = (count / total_sample_decks) * 100 if total_sample_decks > 0 else 0
                archetype_data = {
                    "name": archetype, 
                    "count_in_sample": count, 
                    "sample_percentage": sample_percentage,
                    "note": "This percentage represents prevalence in our sample only, not actual meta share"
                }
                
                if count >= 5:  # More prevalent in sample
                    archetype_tiers["prevalent"].append(archetype_data)
                elif count >= 2:  # Common in sample
                    archetype_tiers["common"].append(archetype_data)
                else:  # Uncommon in sample
                    archetype_tiers["uncommon"].append(archetype_data)
            
            # Sort each tier
            for tier in archetype_tiers:
                if "meta_percentage" in archetype_tiers[tier][0] if archetype_tiers[tier] else False:
                    archetype_tiers[tier].sort(key=lambda x: x["meta_percentage"], reverse=True)
                else:
                    archetype_tiers[tier].sort(key=lambda x: x.get("count_in_sample", 0), reverse=True)
    
    # Check if deck name analysis contains archetype information
    if 'enhanced_meta' in json_files and 'deck_name_analysis' in json_files['enhanced_meta']:
        deck_name_analysis = json_files['enhanced_meta']['deck_name_analysis']
        if 'archetype_stats' in deck_name_analysis:
            # Add this information to characteristics
            archetype_characteristics['name_derived'] = {
                archetype: count for archetype, count in 
                sorted(deck_name_analysis['archetype_stats'].items(), key=lambda x: x[1], reverse=True)
            }
    
    # Extract archetype characteristics from parse_meta if available
    if 'parse_meta' in json_files and 'format_characteristics' in json_files['parse_meta'] and 'archetypes' in json_files['parse_meta']['format_characteristics']:
        archetype_characteristics['descriptions'] = json_files['parse_meta']['format_characteristics']['archetypes']
    
    result = {
        "distribution": archetype_tiers,
        "characteristics": archetype_characteristics
    }
    
    # Add disclaimer based on data source
    if 'meta_representation' in json_files:
        result["source"] = "Actual meta representation data from MTGGoldfish"
    else:
        result["disclaimer"] = "Note: The archetype distribution shown here is based only on our sample decks and does not reflect the actual meta share. For accurate meta share data, refer to sources like MTGGoldfish or tournament results."
    
    return result

def analyze_themes(json_files):
    """Analyze common themes from semantic analysis."""
    themes = {}
    
    # Try sources in order of preference
    for source in ['enhanced_meta', 'semantic_meta']:
        if source in json_files:
            # Extract unigrams and bigrams from meta themes
            if 'meta_themes' in json_files[source]:
                meta_themes = json_files[source]['meta_themes']
                
                # Filter unigrams to meaningful ones
                if 'unigrams' in meta_themes:
                    meaningful_unigrams = [
                        u for u in meta_themes['unigrams'] 
                        if u['score'] > 100 and len(u['term']) > 2  # Higher threshold for relevance
                    ]
                    themes['unigrams'] = meaningful_unigrams[:15]  # Top 15
                
                # Filter bigrams to meaningful ones
                if 'bigrams' in meta_themes:
                    meaningful_bigrams = [
                        b for b in meta_themes['bigrams'] 
                        if b['score'] > 50 and ' ' in b['term']  # Make sure it's actually a bigram
                    ]
                    themes['bigrams'] = meaningful_bigrams[:15]  # Top 15
            
            # Extract any emergent themes from semantic analysis
            if 'synergies' in json_files[source] and 'emergent_themes' in json_files[source]['synergies']:
                themes['emergent'] = json_files[source]['synergies']['emergent_themes']
                
            # If we found themes, no need to check the other source
            if themes:
                break
    
    # Extract themes from deck name analysis if available
    if 'enhanced_meta' in json_files and 'deck_name_analysis' in json_files['enhanced_meta']:
        deck_name_analysis = json_files['enhanced_meta']['deck_name_analysis']
        
        # Extract cross-deck analyses
        if 'cross_deck_analyses' in deck_name_analysis:
            cross_deck_themes = []
            for term, analysis in deck_name_analysis['cross_deck_analyses'].items():
                cross_deck_themes.append({
                    'term': term,
                    'deck_count': analysis['deck_count'],
                    'common_cards': analysis['common_cards'][:5] if 'common_cards' in analysis else [],
                    'common_terms': analysis['common_terms'][:5] if 'common_terms' in analysis else []
                })
            
            if cross_deck_themes:
                themes['name_derived'] = cross_deck_themes
    
    return themes

def extract_mechanics(json_files):
    """Identify most common mechanics and keywords."""
    mechanics = Counter()
    
    # Combine information from multiple sources
    if 'keyword_meta' in json_files and 'format_characteristics' in json_files['keyword_meta'] and 'keywords' in json_files['keyword_meta']['format_characteristics']:
        for keyword, count in json_files['keyword_meta']['format_characteristics']['keywords'].items():
            mechanics[keyword] = max(mechanics[keyword], count)
    
    if 'parse_meta' in json_files and 'format_characteristics' in json_files['parse_meta'] and 'mechanics' in json_files['parse_meta']['format_characteristics']:
        for mechanic, count in json_files['parse_meta']['format_characteristics']['mechanics'].items():
            mechanics[mechanic] = max(mechanics[mechanic], count)
    
    # Extract keywords from deck name analysis if available
    if 'enhanced_meta' in json_files and 'deck_name_analysis' in json_files['enhanced_meta']:
        name_analysis = json_files['enhanced_meta']['deck_name_analysis']
        if 'keyword_stats' in name_analysis:
            for keyword, count in name_analysis['keyword_stats'].items():
                mechanics[keyword] = max(mechanics[keyword], count)
    
    # Convert to list of dictionaries and sort by count
    mechanic_data = [
        {"name": name, "count": count}
        for name, count in mechanics.most_common(20)  # Limit to top 20
    ]
    
    return mechanic_data

def extract_popular_cards(json_files):
    """Compile most played cards across all analyses."""
    card_counts = Counter()
    
    # Combine information from multiple sources
    for source in ['parse_meta', 'keyword_meta']:
        if source in json_files and 'meta_statistics' in json_files[source] and 'most_played_cards' in json_files[source]['meta_statistics']:
            for card_data in json_files[source]['meta_statistics']['most_played_cards']:
                if isinstance(card_data, dict) and 'card' in card_data and 'count' in card_data:
                    card_counts[card_data['card']] += card_data['count']
    
    # Check card frequencies in enhanced_meta or semantic_meta
    for source in ['enhanced_meta', 'semantic_meta']:
        if source in json_files and 'card_frequency' in json_files[source]:
            for card, count in json_files[source]['card_frequency'].items():
                card_counts[card] += count

    # Filter out basic lands
    basic_lands = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest'}
    filtered_cards = [(card, count) for card, count in card_counts.items() if card not in basic_lands]
    
    # Convert to list of dictionaries and sort by count
    card_data = [
        {"name": name, "count": count}
        for name, count in sorted(filtered_cards, key=lambda x: x[1], reverse=True)[:30]  # Limit to top 30
    ]
    
    return card_data

def extract_synergies(json_files):
    """Extract synergy patterns from various analysis methods."""
    synergies = {}
    
    # Check synergies from parse_meta (most traditional pattern-based identification)
    if 'parse_meta' in json_files and 'format_characteristics' in json_files['parse_meta'] and 'synergies' in json_files['parse_meta']['format_characteristics']:
        parse_synergies = json_files['parse_meta']['format_characteristics']['synergies']
        
        # Extract synergy types without going into detail about enablers and payoffs
        # (as these are less reliable than the existence of the synergy itself)
        synergy_types = {}
        for category, synergies_by_type in parse_synergies.items():
            synergy_types[category] = []
            if isinstance(synergies_by_type, dict):
                for synergy_name in synergies_by_type.keys():
                    synergy_types[category].append(synergy_name)
        
        synergies['pattern_based'] = synergy_types
    
    # Check for room synergies specifically
    if 'parse_meta' in json_files and 'format_characteristics' in json_files['parse_meta'] and 'synergies' in json_files['parse_meta']['format_characteristics'] and 'room' in json_files['parse_meta']['format_characteristics']['synergies']:
        room_synergies = json_files['parse_meta']['format_characteristics']['synergies']['room']
        filtered_room_synergies = {
            "unlock_payoffs": room_synergies.get('unlock_payoffs', []),
            "room_types": [
                k for k in room_synergies.keys() 
                if k not in ['unlock_payoffs', 'room_unlock_synergy'] and isinstance(room_synergies[k], list)
            ]
        }
        synergies['room_mechanics'] = filtered_room_synergies
    
    # If enhanced_meta has cross-deck analyses, add this as a synergy source
    if 'enhanced_meta' in json_files and 'deck_name_analysis' in json_files['enhanced_meta'] and 'cross_deck_analyses' in json_files['enhanced_meta']['deck_name_analysis']:
        cross_deck = json_files['enhanced_meta']['deck_name_analysis']['cross_deck_analyses']
        cross_deck_simplified = {}
        
        for term, analysis in cross_deck.items():
            cross_deck_simplified[term] = {
                "deck_count": analysis.get('deck_count', 0),
                "common_cards": analysis.get('common_cards', [])[:5]  # Top 5 cards
            }
        
        synergies['cross_deck'] = cross_deck_simplified
    
    return synergies

def enhance_meta_trends_with_goldfish_data(json_files):
    """Use MTGGoldfish representation data to enhance meta trends."""
    meta_trends = {}
    
    if 'meta_representation' in json_files:
        # Format archetypes by tier and sorted by meta percentage
        archetypes_by_tier = {
            "tier_1": [],  # Top archetypes (5%+ meta)
            "tier_2": [],  # Solid archetypes (1-5% meta)
            "tier_3": []   # Fringe archetypes (<1% meta)
        }
        
        for archetype in json_files['meta_representation']:
            percentage = archetype.get('meta_percentage', 0)
            name = archetype.get('archetype', 'Unknown')
            deck_count = archetype.get('deck_count', 0)
            
            archetype_data = {
                "name": name,
                "meta_percentage": percentage,
                "deck_count": deck_count
            }
            
            if percentage >= 5.0:
                archetypes_by_tier["tier_1"].append(archetype_data)
            elif percentage >= 1.0:
                archetypes_by_tier["tier_2"].append(archetype_data)
            else:
                archetypes_by_tier["tier_3"].append(archetype_data)
        
        # Sort each tier by percentage
        for tier in archetypes_by_tier:
            archetypes_by_tier[tier].sort(key=lambda x: x.get('meta_percentage', 0), reverse=True)
        
        meta_trends["archetype_tiers"] = archetypes_by_tier
        
        # Calculate format diversity metrics
        if archetypes_by_tier["tier_1"]:
            tier1_percentage = sum(a.get('meta_percentage', 0) for a in archetypes_by_tier["tier_1"])
            meta_trends["format_diversity"] = {
                "tier1_count": len(archetypes_by_tier["tier_1"]),
                "tier1_percentage": tier1_percentage,
                "assessment": "High diversity" if tier1_percentage < 25 else 
                              "Medium diversity" if tier1_percentage < 40 else 
                              "Low diversity"
            }
    
    return meta_trends

def generate_meta_report(json_files):
    """Generate a comprehensive meta report from all JSON files."""
    report = {
        "meta_speed": reconcile_meta_speed(json_files),
        "mana_curve_patterns": extract_mana_curve_patterns(json_files),
        "card_type_distributions": analyze_card_type_distributions(json_files),
        "color_combinations": extract_color_combinations(json_files),
        "defining_cards": find_defining_cards(json_files),
        "cluster_patterns": extract_cluster_patterns(json_files),
        "card_clusters": analyze_card_clusters(json_files),
        "card_type_trends": extract_card_type_trends(json_files),
        "archetype_interactions": analyze_archetype_interactions(json_files),
        "themes": analyze_themes(json_files),
        "mechanics": extract_mechanics(json_files),
        "popular_cards": extract_popular_cards(json_files),
        "synergies": extract_synergies(json_files)
    }
    
    # Add MTGGoldfish meta trends if available
    if 'meta_representation' in json_files:
        report["goldfish_meta_trends"] = enhance_meta_trends_with_goldfish_data(json_files)
    
    # Additional meta insights from keyword_meta
    if 'keyword_meta' in json_files and 'format_characteristics' in json_files['keyword_meta']:
        format_chars = json_files['keyword_meta']['format_characteristics']
        report["type_distribution"] = {
            "creature_count": format_chars.get('types', {}).get('creature', 0),
            "instant_count": format_chars.get('types', {}).get('instant', 0),
            "sorcery_count": format_chars.get('types', {}).get('sorcery', 0),
            "enchantment_count": format_chars.get('types', {}).get('enchantment', 0),
            "artifact_count": format_chars.get('types', {}).get('artifact', 0),
            "planeswalker_count": format_chars.get('types', {}).get('planeswalker', 0),
            "land_count": format_chars.get('types', {}).get('land', 0)
        }
    
    # Add insights from deck name analysis if available
    if 'enhanced_meta' in json_files and 'deck_name_analysis' in json_files['enhanced_meta']:
        name_analysis = json_files['enhanced_meta']['deck_name_analysis']
        report["deck_name_patterns"] = {
            "color_identities": name_analysis.get('color_identity_stats', {}),
            "archetypes": name_analysis.get('archetype_stats', {}),
            "keywords": name_analysis.get('keyword_stats', {}),
            "subtypes": name_analysis.get('subtype_stats', {}),
            "card_references": {k: v for k, v in name_analysis.get('card_reference_stats', {}).items() if v >= 3}
        }
    
    return report

def main():
    """Main function to process the JSON files and generate a report."""
    try:
        # Load the JSON files with proper error handling
        json_files = {}
        
        # Standard analysis files
        file_mappings = {
            'parse_meta': 'json_outputs/parse_meta_analysis_results.json',
            'semantic_meta': 'json_outputs/semantic_meta_analysis_results.json',
            'keyword_meta': 'json_outputs/meta_keyword_analysis_results.json',
            'enhanced_meta': 'json_outputs/enhanced_semantic_meta_analysis.json',
            'meta_representation': 'json_outputs/deck_meta_representation.json'
        }
        
        # Load each file with proper error handling
        for key, filepath in file_mappings.items():
            json_files[key] = load_json_file(filepath)
            
            # Check if file was successfully loaded
            if not json_files[key]:
                print(f"Warning: {filepath} was not loaded or is empty.")
            else:
                print(f"Successfully loaded {filepath}.")
        
        # Generate the meta report
        meta_report = generate_meta_report(json_files)
        
        # Write the report to a JSON file
        with open('json_outputs/consolidated_meta_report.json', 'w') as file:
            json.dump(meta_report, file, indent=2)
        
        print("Meta analysis complete! Report written to consolidated_meta_report.json")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()