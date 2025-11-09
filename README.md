# Sensory-CoKGE: Knowledge Graph Embedding for Food Sensory Attributes

This repository supports the published paper:

**"Sensory-CoKGE: A Contextualized Knowledge Graph Embedding Framework Using Language Models for Converting Text-Based Food Attributes into Numerical Representation"**

Published in: *Expert Systems with Applications*, Volume 299, Part C, 2026

[Read the paper](https://doi.org/10.1016/j.eswa.2025.130191)

## Overview

Sensory-CoKGE converts descriptive food sensory text (e.g., "nutty", "acidic", "smooth") into numerical vector representations called embeddings. These embeddings enable **statistical analysis of sensory attributes** - a crucial need in food science and sensory analysis where understanding how attributes relate to and vary with each other is more important than generating text.

By using pretrained language models, the framework captures semantic meaning in sensory descriptors, creating embeddings suitable for clustering, correlation analysis, dimensionality reduction, and other statistical methods.

## Key Features

- **Easy-to-use API** for food researchers with minimal Python experience
- Converts text-based food sensory attributes into numerical embeddings
- **Flexible graph creation** from simple dictionaries or CSV files
- Supports multiple language models (BERT, RoBERTa, ALBERT, GPT-2, BART, T5)
- Optional large language models (Gemma2, Llama3, Qwen2) for advanced analysis
- **Customizable context templates** following pattern: "This [Food] [Verb] [Attribute]"
- Built-in validation and evaluation metrics
- Works with any food product (wine, cheese, coffee, chocolate, etc.)

## Prerequisites & Installation

### System Requirements

- **Python:** 3.8 or higher
- **Hardware:** Modern computer with internet connection
- **GPU:** Optional but recommended for faster processing
- **CPU-only mode:** Fully supported

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/b05611038/sensory-cokge.git
   cd sensory-cokge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python3 -c "import torch; import transformers; print('Installation successful!')"
   ```

## Quick Start for Food Researchers (5 minutes)

The easiest way to get started is using the user-friendly API. Here's a complete example for analyzing wine attributes:

```python
from sensory_cokge import (
    build_graph_from_hierarchy,
    create_context_template,
    compute_embeddings,
    embeddings_to_csv,
    evaluate_embeddings,
    validate_graph_structure
)

# 1. Define your food attributes as a simple dictionary
wine_attributes = {
    'fruity': ['apple', 'pear', 'citrus', 'berry'],
    'floral': ['rose', 'violet', 'jasmine'],
    'spicy': ['pepper', 'cinnamon', 'clove'],
    'earthy': ['mushroom', 'soil', 'forest']
}

# 2. Build the attribute graph
graph = build_graph_from_hierarchy(wine_attributes, root='wine')

# 3. Validate your graph structure
validation = validate_graph_structure(graph)
print(validation['issues'])  # Check for any problems

# 4. Create appropriate context for your food
context = create_context_template('wine', 'has', '{0} {1} aroma')
# Result: "This wine has {0} {1} aroma."

# 5. Generate embeddings
attributes = [d for d in graph.descriptions if d != 'wine']
embeddings = compute_embeddings(
    attributes,
    model_name='BERT',
    context=context,
    device='auto'
)

# 6. Evaluate how well embeddings preserve graph structure
results = evaluate_embeddings(embeddings, graph=graph)
print(f"Distance Matching Score: {results['distances_matching_l2']:.4f}")

# 7. Export to CSV for analysis
embeddings_to_csv(embeddings, 'wine_embeddings.csv')
print("Done! Check wine_embeddings.csv")
```

**For detailed examples with other foods (cheese, chocolate, coffee), see [GUIDE_FOR_FOOD_RESEARCHERS.ipynb](GUIDE_FOR_FOOD_RESEARCHERS.ipynb)**

## User-Friendly Helper Functions

### Build Graphs from Dictionaries

```python
from sensory_cokge import build_graph_from_hierarchy

# Simple nested dictionary
cheese_flavors = {
    'dairy': {
        'fresh': ['milk', 'cream', 'butter'],
        'aged': ['sharp', 'tangy']
    },
    'savory': ['umami', 'salty'],
    'pungent': ['funky', 'ammonia']
}

graph = build_graph_from_hierarchy(cheese_flavors, root='cheese')
```

### Build Graphs from CSV Files

```python
from sensory_cokge import build_graph_from_csv

# CSV format:
# attribute,category
# fruity,root
# apple,fruity
# pear,fruity

graph = build_graph_from_csv('my_attributes.csv',
                              child_column='attribute',
                              parent_column='category',
                              root='root')
```

### Create Context Templates

```python
from sensory_cokge import create_context_template

# Pattern: "This [Food] [Verb] [Attribute]."

wine_context = create_context_template('wine', 'has', '{0} {1} aroma')
# → "This wine has {0} {1} aroma."

cheese_context = create_context_template('cheese', 'tastes', '{0} {1}')
# → "This cheese tastes {0} {1}."

coffee_context = create_context_template('coffee', 'smells', '{0} {1}')
# → "This coffee smells {0} {1}."
```

### Validate Graph Structure

```python
from sensory_cokge import validate_graph_structure

validation = validate_graph_structure(graph)

if validation['is_valid']:
    print("✓ Graph is valid!")
else:
    print("Issues:", validation['issues'])

print(f"Attributes: {validation['statistics']['num_descriptions']}")
print(f"Connections: {validation['statistics']['num_connections']}")
```

## Advanced Workflow: Using Scripts Directly

For reproducibility and batch processing, you can use the provided scripts:

### Workflow A: Pretrained Models (Recommended)

```bash
# Step 1: Generate embeddings from pretrained models
python3 embeddings_from_pretrained.py

# Step 2: Evaluate model performance
python3 models_evaluation.py pretrained

# Step 3 (Optional): Visualize embeddings
jupyter notebook Notebook_embedding_visualization.ipynb
```

### Workflow B: Fine-tuning Models

```bash
# Step 1: Generate synthetic training data
python3 generate_finetuned_data.py

# Step 2: Fine-tune a model
python3 finetune_BERT_by_sequence_classification.py <config_file>

# Step 3: Generate embeddings from fine-tuned model
python3 embeddings_from_finetuned.py <model_name>

# Step 4: Evaluate fine-tuned model
python3 models_evaluation.py finetuned
```

## Available Models

### Small Models (Fine-tunable)

These models can be evaluated and fine-tuned on standard computers.

| Model | Parameters | Best For |
|-------|-----------|----------|
| **BERT** | 110M | Quick baseline, general use |
| **RoBERTa** | 355M | Better accuracy, production |
| **ALBERT** | 12M | Limited resources, fastest |
| **GPT-2** | 124M | Generative approach |
| **BART** | 406M | Sequence tasks |
| **T5** | 220M | Flexible architecture |

**Recommendation:** Start with **BERT** for quick testing, then try **RoBERTa** for better results.

### Large Models (Evaluation Only)

These models provide state-of-the-art embeddings but **cannot be fine-tuned** with this codebase.

| Model | Parameters | Notes |
|-------|-----------|-------|
| **Gemma2** | 8B | Requires substantial GPU memory |
| **Llama3** | 8B | Requires substantial GPU memory |
| **Qwen2** | 8B | Requires substantial GPU memory |

```bash
# Enable large model evaluation (optional)
python3 embeddings_from_pretrained.py --enable_LLM
```

## Adapting to Your Food Product

The framework is designed to work with **any food product**. Here are complete examples:

### Example 1: Wine Analysis
```python
wine_attributes = {
    'fruity': {
        'citrus': ['lemon', 'lime', 'grapefruit'],
        'stone_fruit': ['peach', 'apricot'],
        'berry': ['strawberry', 'raspberry']
    },
    'floral': ['rose', 'violet', 'jasmine'],
    'spicy': ['pepper', 'cinnamon', 'clove']
}
graph = build_graph_from_hierarchy(wine_attributes, root='wine')
context = create_context_template('wine', 'has', '{0} {1} aroma')
```

### Example 2: Chocolate Tasting
```python
chocolate_profile = {
    'sweet': ['honey', 'caramel', 'vanilla'],
    'bitter': ['cocoa', 'dark', 'roasted'],
    'fruity': {
        'berry': ['raspberry', 'cherry'],
        'citrus': ['orange', 'lemon']
    },
    'nutty': ['almond', 'hazelnut']
}
graph = build_graph_from_hierarchy(chocolate_profile, root='chocolate')
context = create_context_template('chocolate', 'has', '{0} {1} taste')
```

### Example 3: Coffee Flavor Wheel
```python
# Using CSV file
# coffee_attributes.csv:
# attribute,category
# fruity,root
# berry,fruity
# citrus,fruity
# nutty,root
# chocolate,root

graph = build_graph_from_csv('coffee_attributes.csv', root='root')
context = create_context_template('coffee', 'has', '{0} {1} flavor')
```

**See [GUIDE_FOR_FOOD_RESEARCHERS.ipynb](GUIDE_FOR_FOOD_RESEARCHERS.ipynb) for more examples including cheese, beer, bread, and olive oil.**

## Complete Workflow Examples

### Example: Evaluating Embeddings with Custom Graphs

After generating embeddings for your custom food graph, evaluate how well they preserve the graph structure:

```python
from sensory_cokge import (
    build_graph_from_hierarchy,
    create_context_template,
    compute_embeddings,
    evaluate_embeddings
)

# 1. Build custom wine graph
wine_attributes = {
    'fruity': ['apple', 'pear', 'citrus', 'berry'],
    'floral': ['rose', 'violet', 'jasmine'],
    'spicy': ['pepper', 'cinnamon', 'clove']
}
wine_graph = build_graph_from_hierarchy(wine_attributes, root='wine')

# 2. Generate embeddings
context = create_context_template('wine', 'has', '{0} {1} aroma')
attributes = [d for d in wine_graph.descriptions if d != 'wine']
embeddings = compute_embeddings(attributes, model_name='BERT', context=context)

# 3. Evaluate embeddings against your custom graph
results = evaluate_embeddings(embeddings, graph=wine_graph)

# 4. Check results
print(f"L2 Distance Matching: {results['distances_matching_l2']:.4f}")
print(f"Angular Distance Matching: {results['distances_matching_angle']:.4f}")
print(f"Adjacency Matching: {results['adjacency_matching_l2']:.4f}")

# Higher scores = better preservation of graph structure
```

**Note:** If you don't provide a `graph` parameter, `evaluate_embeddings()` defaults to the coffee flavor wheel.

### Example: Generating Synthetic Training Data for Custom Foods

Create training data for fine-tuning models on your specific food domain:

```python
from sensory_cokge import (
    build_graph_from_hierarchy,
    generate_synthetic_data
)

# 1. Build custom cheese graph
cheese_flavors = {
    'dairy': {
        'fresh': ['milk', 'cream', 'butter'],
        'aged': ['sharp', 'tangy']
    },
    'savory': ['umami', 'salty'],
    'pungent': ['funky', 'ammonia']
}
cheese_graph = build_graph_from_hierarchy(cheese_flavors, root='cheese')

# 2. Generate synthetic training data (food_name auto-detected from root!)
data = generate_synthetic_data(
    train_samples=10000,
    eval_samples=1000,
    graph=cheese_graph,           # food_name will auto-detect as 'cheese'
    output_dir='./cheese_training',
    save_csv=True
)

print(f"Generated {len(data['train'])} training samples")
print(f"Generated {len(data['eval'])} evaluation samples")
print(f"Files saved to ./cheese_training/")

# 3. Optional: Override auto-detected food name for more specific naming
wine_data = generate_synthetic_data(
    train_samples=5000,
    eval_samples=500,
    graph=wine_graph,
    food_name='red wine',  # Override to be more specific
    output_dir='./wine_training'
)

# 4. Use the generated data to fine-tune a model
# See finetune_BERT_by_sequence_classification.py for details
```

**Smart food name detection:** When you provide a custom `graph`, the `food_name` is automatically detected from the graph's root name. For example:
- `build_graph_from_hierarchy(..., root='wine')` → Generated text uses "wine"
- `build_graph_from_hierarchy(..., root='cheese')` → Generated text uses "cheese"
- `build_graph_from_hierarchy(..., root='chocolate')` → Generated text uses "chocolate"

**Default behavior:** If you don't provide a `graph` parameter, it generates data for the coffee flavor wheel.

**Custom override:** You can still explicitly set `food_name` to override the auto-detection (e.g., `food_name='red wine'` instead of just 'wine').

## Understanding Your Results

### Evaluation Metrics

The framework computes metrics showing how well embeddings preserve relationships:

- **Adjacency Matching:** Are connected descriptors nearby in embedding space?
- **Distance Matching:** Do embedding distances match graph distances?
- **L2 vs Angular:** Different geometric measures of embedding quality

Higher scores indicate better preservation of sensory attribute structure.

### Using Embeddings for Analysis

Once you have embeddings, you can use standard data science tools:

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load embeddings
df = pd.read_csv('wine_embeddings.csv')

# Perform PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(df.iloc[:, 1:])  # Skip description column

# Cluster similar attributes
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(df.iloc[:, 1:])

# Statistical analysis
# correlation, t-tests, ANOVA, etc.
```

## Troubleshooting

### Module Not Found

```bash
ModuleNotFoundError: No module named 'sensory_cokge'
```

**Solution:** Make sure you're in the repository directory and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Out of Memory

If you encounter memory errors:

1. Use a smaller model (BERT or ALBERT)
2. Reduce batch size:
   ```bash
   python3 embeddings_from_pretrained.py --batch_size 2
   ```
3. Use CPU mode:
   ```bash
   python3 embeddings_from_pretrained.py --device cpu
   ```

### Graph Validation Errors

```python
validation = validate_graph_structure(graph)
if not validation['is_valid']:
    print("Issues:", validation['issues'])
    # Common issues:
    # - Cycles in graph (circular references)
    # - Unreachable nodes (disconnected attributes)
    # - Missing root node
```

Fix by ensuring:
- All attributes connect back to root
- No circular references (A→B→C→A)
- Proper parent-child relationships

### First Run is Slow

The first run downloads models from the internet. This is normal and only happens once:
- Small models (BERT): ~500MB download
- Large models (optional): ~5GB each
- Models are cached locally for future use

## Files and Structure

**Main Scripts:**
- `embeddings_from_pretrained.py` - Generate pretrained embeddings
- `embeddings_from_finetuned.py` - Generate fine-tuned embeddings
- `models_evaluation.py` - Evaluate and compare embeddings
- `generate_finetuned_data.py` - Create training data
- `finetune_[MODEL]_by_sequence_classification.py` - Fine-tune models

**Notebooks:**
- `Notebook_embedding_visualization.ipynb` - Visualize embeddings with t-SNE/PCA
- `Notebook_OOD_visualization.ipynb` - Out-of-distribution analysis

**Documentation:**
- `GUIDE_FOR_FOOD_RESEARCHERS.ipynb` - **Interactive guide for food researchers with examples**
- `README.md` - This file

**Source Code:**
- `sensory_cokge/` - Main package
  - `graph.py` - Description graph and knowledge graph functions
  - `models.py` - Language model implementations
  - `metrics.py` - Evaluation metrics
  - `finetune.py` - Synthetic data generation
  - `utils.py` - Utility functions
  - `__init__.py` - User-friendly API and helper functions

## Core API Reference

### Graph Creation
- `build_graph_from_hierarchy(hierarchy, root, graph_name)` - Create graph from nested dictionary
- `build_graph_from_csv(filepath, child_column, parent_column, root)` - Create graph from CSV file
- `create_description_graph(descriptions, connections, root)` - Create graph from explicit connections
- `validate_graph_structure(graph)` - Validate graph is a proper DAG

### Embedding Generation
- `compute_embeddings(descriptions, model_name, context, device, ...)` - Generate embeddings from descriptions
- `embeddings_to_csv(embeddings, filepath)` - Export embeddings as CSV
- `evaluate_embeddings(embeddings, graph, embedding_type)` - Evaluate structural consistency against a graph (supports custom graphs)

### Context Templates
- `create_context_template(food_name, verb, attribute_placeholder)` - Generate context following "This [Food] [Verb] [Attribute]" pattern

### Synthetic Data
- `generate_synthetic_data(train_samples, eval_samples, graph, food_name, ...)` - Create training data from graph (supports custom graphs for any food; food_name auto-detects from graph root)

## FAQ

**Q: I'm a food researcher with limited Python experience. Where should I start?**

A: Read [GUIDE_FOR_FOOD_RESEARCHERS.ipynb](GUIDE_FOR_FOOD_RESEARCHERS.ipynb) and follow the Quick Start example above. You only need to modify the dictionary with your food's attributes.

**Q: Can I use this for foods other than coffee?**

A: Yes! The framework works for **any food product**. Examples are provided for wine, cheese, chocolate, coffee, beer, bread, and olive oil. Simply define your food's sensory attributes and create the appropriate context.

**Q: What's the difference between pretrained and fine-tuned models?**

A: Pretrained models work immediately but are general-purpose. Fine-tuning adapts them to your specific domain, potentially improving results for your food category.

**Q: How do I define a valid attribute graph?**

A: Your graph must be a DAG (Directed Acyclic Graph):
- Start with one root (your food name)
- Build hierarchically (categories → sub-categories → attributes)
- No cycles (A→B→C→A is invalid)
- Use `validate_graph_structure()` to check

**Q: Can I modify the context template?**

A: Yes! Use `create_context_template()` with different verbs:
- Taste: `create_context_template('cheese', 'tastes', '{0} {1}')`
- Smell: `create_context_template('coffee', 'smells', '{0} {1}')`
- Feel: `create_context_template('bread', 'feels', '{0} {1}')`

**Q: Where are results saved?**

A: All embeddings and models go to `./outputs/` directory by default. CSV exports go to your specified filename.

**Q: Can I fine-tune large models (Gemma2, Llama3, Qwen2)?**

A: No, this codebase only provides fine-tuning for small models (BERT, RoBERTa, ALBERT, GPT-2, BART, T5). Large models are for evaluation only.

## Citation

If you use Sensory-CoKGE in your research, please cite:

```bibtex
@article{chang2026sensory-cokge,
  title = {Sensory-CoKGE: A contextualized knowledge graph embedding framework using language models for converting text-based food attributes into numerical representation},
  journal = {Expert Systems with Applications},
  volume = {299},
  number = {Part C},
  year = {2026},
  issn = {0957-4174},
  doi = {10.1016/j.eswa.2025.130191},
  url = {https://doi.org/10.1016/j.eswa.2025.130191},
  author = {Yu-Tang Chang and Shih-Fang Chen},
}
```

## License

See LICENSE file for details.

## Contributing

This repository is maintained to support the published research. For questions or issues, please open a GitHub issue.
