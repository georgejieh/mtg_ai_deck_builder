# MTG AI Deck Builder
_Unleash the power of **Machine Learning** to forge next-level **Magic: The Gathering** decks that adapt to the ever-evolving **Standard** meta!_

[![Magic: The Gathering](https://img.shields.io/badge/Magic%3A%20the%20Gathering-AI%20Deck%20Builder-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](#)
[![Unsupervised Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-green)](#)

## Overview
**MTG AI Deck Builder** aims to train a **dynamic**, unsupervised **AI model** that consistently updates itself to generate competitive **Standard** decks. By tapping into the **Scryfall** API for card data, it analyzes existing decks and the broader meta—ultimately discovering creative synergies and archetype strategies that might be overlooked by the community.

### Why This Project?
- **Adaptive Metagame Analysis**: Stay ahead of format shifts by re-training on newly released sets or emergent archetypes.
- **Unsupervised Creativity**: Rely on an unsupervised learning framework to find unexpected or “under-the-radar” combinations.
- **Data-Driven**: Harness card data from Scryfall, letting the model continuously refine deck lists as the meta evolves.
- **Solo & Open-Source**: This is primarily a solo project, but feel free to fork and experiment on your own!

## Current Status
> :warning: **Work in progress**: The current scripts provide only basic, “cookie-cutter” analyses.

- **[`fetch_standard_legal_cards.py`]**  
  Fetches all Standard-legal cards from the Scryfall API and outputs them as a CSV dataset.
- **[`deck_analysis.py`]**  
  Analyzes an input deck list to produce baseline archetype insights. Uses very simplified guidelines, so suggestions may not be fully accurate yet.
- **[`analyze_meta.py`]**  
  Explores decks that have more than 0.5% play, applying a similarly basic outline to identify **interactions**, **keywords**, and **synergies**. Future enhancements will improve detection of newly introduced mechanics.

## Key Features
1. **Scryfall Integration**  
   Automatically pulls the latest **Standard**-legal cards, ensuring the model is always up-to-date.
2. **Deck Archetype Analysis**  
   Categorizes decks into established archetypes (Aggro, Midrange, Control, Tempo, Combo) using simplified heuristics.
3. **Meta Analysis**  
   Aggregates data from decks above 0.5% play to highlight **format trends** and synergy clusters.
4. **Unsupervised Learning Potential**  
   Plans to integrate an AI model that **auto-generates** decklists—unconstrained by conventional archetype thinking.

## Installation
Since this project is still under active development, an official list of dependencies (`requirements.txt`) isn’t available yet. You’ll likely need:
- **Python 3.x**
- Common data libraries like **pandas**, **numpy**, **requests**, etc.

Clone the repo:
```bash
git clone https://github.com/georgejieh/mtg_ai_deck_builder.git
cd mtg_ai_deck_builder
```
Then install any libraries that come up as you test the scripts (e.g., `pip install pandas requests`).

## Usage Example

### 1. Fetch Standard-Legal Cards
Pull down all Standard-legal cards (saves `standard_cards.csv` to `./data`):
```bash
python fetch_standard_legal_cards.py
```

### 2. Analyze a Deck
To analyze a single deck list, place your deck in a `.txt` file and run:
```bash
python deck_analysis.py /path/to/decklist.txt
```

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

## Roadmap
- **Enhance Archetype Logic**  
  Incorporate more nuanced synergy/keyword detection as new sets release.
- **Automated Meta Updates**  
  Improve scripts to detect new mechanics automatically, providing real-time meta insights.
- **Neural Network Model**  
  Implement an unsupervised (possibly semi-supervised) approach to **auto-generate** innovative decklists.
- **Self-Training**  
  Continually retrain as new sets and meta changes arise, refining synergy detection beyond current “cookie-cutter” methods.
- **User Interface**  
  Explore a simple web-based front-end for deck analysis and meta breakdown.

## License
This project is available under the [MIT License](LICENSE). Since this is a solo project, no external contributions are expected—but feel free to fork and experiment.

---

Stay tuned for continuous updates as the **AI Deck Builder** project evolves—pushing the boundaries of MTG tech, one set at a time!