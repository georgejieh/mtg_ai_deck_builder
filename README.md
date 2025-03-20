# MTG AI Deck Builder
_Unleash the power of **Machine Learning** to forge next-level **Magic: The Gathering** decks that adapt to the ever-evolving **Standard** meta!_

[![Magic: The Gathering](https://img.shields.io/badge/Magic%3A%20the%20Gathering-AI%20Deck%20Builder-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](#)
[![Unsupervised Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-green)](#)

## Overview
**MTG AI Deck Builder** aims to train a **dynamic**, unsupervised **AI model** that consistently updates itself to generate competitive **Standard** decks. By tapping into the **Scryfall** API for card data, it analyzes existing decks and the broader meta—ultimately discovering creative synergies and archetype strategies that might be overlooked by the community.

### Why This Project?
- **Adaptive Metagame Analysis**: Stay ahead of format shifts by re-training on newly released sets or emergent archetypes.
- **Unsupervised Creativity**: Rely on an unsupervised learning framework to find unexpected or "under-the-radar" combinations.
- **Data-Driven**: Harness card data from Scryfall, letting the model continuously refine deck lists as the meta evolves.
- **Solo & Open-Source**: This is primarily a solo project, but feel free to fork and experiment on your own!

## Current Status
> :warning: **Work in progress**: The project features multiple analysis approaches with varying degrees of sophistication.

- **[`fetch_standard_legal_cards.py`]**  
  Fetches all Standard-legal cards from the Scryfall API and outputs them as a CSV dataset. Handles various card layouts including split cards and Room cards, and extracts comprehensive card data including types, colors, keywords, and mechanics.

- **[`deck_analysis.py`]**  
  Analyzes an input deck list to produce baseline archetype insights. Uses statistical analysis to determine if a deck fits Aggro, Midrange, Control, Tempo, or Combo archetypes, and provides detailed breakdowns of mana curve, card compositions, and mechanic distributions.

- **[`current_standard_deck_list_scraper.py`]**  
  Scrapes the latest Standard deck lists from MTGGoldfish's metagame page. Allows filtering by minimum meta percentage and saves deck lists as text files for analysis. Also exports meta representation data as JSON.

- **[`analyze_meta_old_try_to_parse.py`]**  
  The first meta analysis script that uses rule-based pattern matching to identify mechanics and synergies. It parses oracle text using predefined regex patterns to detect card interactions, with special handling for Room-type enchantments. While comprehensive, its accuracy depends on the quality of the predefined patterns. It can misclassify complex interactions or miss new mechanics that don't match its patterns.

- **[`analyze_meta_using_keywords.py`]**  
  A statistical approach that analyzes card data directly without assumptions about interactions. It dynamically extracts types, subtypes, keywords, and references from the card pool to identify patterns. This approach offers more reliable results when the card pool changes, as it doesn't rely on hardcoded patterns. It excels at providing objective meta statistics but offers less insight into complex card interactions.

- **[`semantics_meta_analysis.py`]**  
  The newest script that implements machine learning and semantic analysis. It uses a pre-trained sentence transformer model to generate embeddings for cards based on their oracle text. It then applies clustering techniques to identify similar cards and decks without relying on predefined patterns. This approach can discover nuanced relationships and emergent themes that might be missed by rule-based systems, though the identified similarities may sometimes lack clear explanation since the model isn't specifically trained on Magic terminology.

- **[`consolidated_meta_analysis.py`]**  
  Combines the outputs from all three meta analysis approaches (pattern-based, keyword-based, and semantic) to generate a comprehensive meta report. It reconciles potentially conflicting information from different analysis methods, extracts the most reliable insights from each, and produces a unified view of the metagame including archetype distributions, card type trends, color combinations, and synergy clusters.

All three meta analysis scripts are maintained in the repository as they provide complementary insights for different purposes. Use the pattern-matching approach for detailed mechanic breakdowns, the keyword-based approach for reliable statistical analysis, and the semantic approach for discovering unexpected card relationships.

## Key Features
1. **Scryfall Integration**  
   Automatically pulls the latest **Standard**-legal cards, ensuring the model is always up-to-date.
2. **Deck Archetype Analysis**  
   Categorizes decks into established archetypes (Aggro, Midrange, Control, Tempo, Combo) using multiple approaches:
   - Statistical analysis of card distributions
   - Pattern matching on card mechanics
   - Semantic similarity clustering
3. **Meta Analysis**  
   Multiple approaches to analyze the metagame:
   - Pattern-based mechanic and synergy detection
   - Statistical keyword and type analysis
   - Machine learning-based semantic analysis
4. **Unsupervised Learning Potential**  
   Plans to integrate an AI model that **auto-generates** decklists—unconstrained by conventional archetype thinking.

## Installation
Since this project is still under active development, an official list of dependencies (`requirements.txt`) isn't available yet. You'll likely need:
- **Python 3.x**
- Common data libraries like **pandas**, **numpy**, **requests**, etc.
- For semantic analysis: **sentence-transformers**, **scikit-learn**

Clone the repo:
```bash
git clone https://github.com/georgejieh/mtg_ai_deck_builder.git
cd mtg_ai_deck_builder
```
Then install any libraries that come up as you test the scripts (e.g., `pip install pandas requests sentence-transformers scikit-learn`).

## Usage Example

### 1. Fetch Standard-Legal Cards
Pull down all Standard-legal cards (saves `standard_cards.csv` to `./data`):
```bash
python fetch_standard_legal_cards.py
```

### 2. Scrape Current Meta Decks
Scrape the latest Standard deck lists from MTGGoldfish:
```bash
python current_standard_deck_list_scraper.py
```

### 3. Analyze a Deck
To analyze a single deck list, place your deck in a `.txt` file and run:
```bash
python deck_analysis.py /path/to/decklist.txt
```

### 4. Analyze the Meta
You can use any of the three meta analysis scripts based on your needs:
```bash
# For rule-based pattern matching (comprehensive but potentially less accurate):
python analyze_meta_old_try_to_parse.py --cards data/standard_cards.csv --decks current_standard_decks

# For statistical keyword-based analysis (more reliable but less insightful):
python analyze_meta_using_keywords.py --cards data/standard_cards.csv --decks current_standard_decks

# For semantic analysis using machine learning (discovers nuanced relationships):
python semantics_meta_analysis.py --cards data/standard_cards.csv --decks current_standard_decks
```

### 5. Generate Consolidated Meta Report
Combine insights from all three meta analysis approaches:
```bash
python consolidated_meta_analysis.py
```

Each script will generate its own analysis output file and display a summary report in the console.

#### Decklist Format
The **mainboard** and **sideboard** are separated by a blank line. The script only analyzes the mainboard. For instance:

<details>
<summary>Sample Deck List</summary>

```
2 Anoint with Affliction
3 Cut Down
4 Darkslick Shores
4 Deep-Cavern Bat
3 Enduring Curiosity
4 Floodpits Drowner
1 Fountainport
4 Gloomlake Verge
2 Go for the Throat
4 Island
3 Kaito, Bane of Nightmares
4 Mockingbird
4 Oildeep Gearhulk
4 Preacher of the Schism
4 Restless Reef
1 Shoot the Sheriff
4 Swamp
1 Three Steps Ahead
4 Underground River

4 Duress
2 Feed the Swarm
2 Ghost Vacuum
1 Gix's Command
1 Malicious Eclipse
2 The Filigree Sylex
2 Tishana's Tidebinder
1 Withering Torment
```

</details>

The **blank line** between `4 Underground River` and `4 Duress` indicates where the **mainboard** ends and the **sideboard** begins.

## Comparison of Meta Analysis Approaches

| Feature | Pattern-Based | Keyword-Based | Semantic Analysis |
|---------|--------------|---------------|-------------------|
| **Approach** | Rule-based with regex patterns | Statistical analysis of card data | Machine learning with text embeddings |
| **Strengths** | Detailed mechanic breakdown<br>Synergy identification<br>Room card handling | Reliable with changing card pools<br>Objective meta statistics<br>No assumptions needed | Discovers nuanced relationships<br>Finds emergent themes<br>Not limited by predefined patterns |
| **Limitations** | May miss new mechanics<br>Pattern accuracy depends on rules<br>Less adaptable | Less insight into card interactions<br>Limited synergy detection<br>More descriptive than analytical | Less interpretable results<br>Model not trained on Magic terminology<br>Requires additional dependencies |
| **Best For** | Mechanic & synergy analysis<br>Room card interactions<br>Detailed breakdown | Objective meta statistics<br>Format speed analysis<br>Reliable archetype detection | Discovering unexpected relationships<br>Deck clustering<br>Finding hidden patterns |

## Roadmap
- **Enhance Archetype Logic**  
  Incorporate more nuanced synergy/keyword detection as new sets release.
- **Automated Meta Updates**  
  Improve scripts to detect new mechanics automatically, providing real-time meta insights.
- **Neural Network Model**  
  Implement an unsupervised (possibly semi-supervised) approach to **auto-generate** innovative decklists.
- **Self-Training**  
  Continually retrain as new sets and meta changes arise, refining synergy detection beyond current methods.
- **User Interface**  
  Explore a simple web-based front-end for deck analysis and meta breakdown.
- **Train Custom Embeddings**  
  Develop Magic-specific word embeddings to improve the semantic analysis accuracy.

## License
This project is available under the [MIT License](LICENSE). Since this is a solo project, no external contributions are expected—but feel free to fork and experiment.

---

Stay tuned for continuous updates as the **AI Deck Builder** project evolves—pushing the boundaries of MTG tech, one set at a time!