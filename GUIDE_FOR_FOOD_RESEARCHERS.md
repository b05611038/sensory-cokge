# Guide for Food Researchers

This guide helps food researchers adapt `sensory_cokge` to analyze sensory attributes of any food product, even with minimal Python experience.

## Quick Start: 3 Steps to Analyze Your Food

### Step 1: Define Your Attribute Graph

Choose the easiest method for your workflow:

#### Method A: Python Dictionary (Recommended for Beginners)

```python
from sensory_cokge import build_graph_from_hierarchy

# Define your food's attributes as a nested dictionary
wine_attributes = {
    'fruity': ['apple', 'pear', 'citrus', 'berry'],
    'floral': ['rose', 'violet', 'jasmine'],
    'spicy': ['pepper', 'cinnamon', 'clove'],
    'earthy': ['mushroom', 'soil', 'forest']
}

# Build the graph
graph = build_graph_from_hierarchy(
    wine_attributes,
    root='wine',
    graph_name='wine_aromas'
)
```

#### Method B: CSV File (Recommended for Excel Users)

Create a CSV file with your attributes:

```csv
attribute,category
fruity,root
earthy,root
spicy,root
apple,fruity
pear,fruity
citrus,fruity
mushroom,earthy
soil,earthy
```

Then load it:

```python
from sensory_cokge import build_graph_from_csv

graph = build_graph_from_csv(
    'wine_attributes.csv',
    child_column='attribute',
    parent_column='category',
    root='root',
    graph_name='wine_aromas'
)
```

### Step 2: Create Context Template

Tell the model how to describe your food:

```python
from sensory_cokge import create_context_template

# Pattern: "This [Food] [Verb] [Attribute]."
context = create_context_template('wine', 'has', '{0} {1} aroma')
# Result: "This wine has {0} {1} aroma."

# Other examples:
# Cheese: create_context_template('cheese', 'tastes', '{0} {1}')
# Coffee: create_context_template('coffee', 'smells', '{0} {1}')
# Bread: create_context_template('bread', 'feels', '{0} {1}')
```

### Step 3: Compute Embeddings

```python
from sensory_cokge import compute_embeddings, embeddings_to_csv

# Get all attributes except root
attributes = [desc for desc in graph.descriptions if desc != 'wine']

# Compute embeddings
embeddings = compute_embeddings(
    attributes,
    model_name='BERT',  # or 'RoBERTa', 'GPT2', etc.
    context=context,
    device='auto'
)

# Export to CSV for analysis
embeddings_to_csv(embeddings, 'wine_embeddings.csv')
```

## How to Define Valid Attribute Graphs

### Graph Rules (DAG - Directed Acyclic Graph)

Your attribute graph must follow these rules:

1. **Start with one root** - Your food product name
2. **Build hierarchically** - From general to specific
3. **No cycles** - Never loop back

#### ✅ Valid Structure

```
wine (root)
├── fruity
│   ├── apple
│   ├── pear
│   └── citrus
└── spicy
    ├── pepper
    └── cinnamon
```

#### ❌ Invalid Structure (has cycle)

```
wine → fruity → apple → fruity  ← CYCLE! Not allowed!
```

### Python Dictionary Format

```python
# Simple structure (2 levels)
simple_hierarchy = {
    'category1': ['attr1', 'attr2', 'attr3'],
    'category2': ['attr4', 'attr5']
}

# Nested structure (3+ levels)
nested_hierarchy = {
    'main_category': {
        'sub_category1': ['attr1', 'attr2'],
        'sub_category2': ['attr3', 'attr4']
    },
    'another_main': ['attr5', 'attr6']
}
```

### CSV Format

```csv
attribute,category
main_category,root
sub_category,main_category
specific_attr,sub_category
```

## Complete Examples by Food Type

### Example 1: Wine Sensory Analysis

```python
from sensory_cokge import (
    build_graph_from_hierarchy,
    create_context_template,
    compute_embeddings,
    embeddings_to_csv,
    validate_graph_structure
)

# 1. Define wine attributes
wine_attributes = {
    'fruity': {
        'citrus': ['lemon', 'lime', 'grapefruit', 'orange'],
        'stone_fruit': ['peach', 'apricot', 'plum'],
        'berry': ['strawberry', 'raspberry', 'blackberry']
    },
    'floral': ['rose', 'violet', 'jasmine', 'honeysuckle'],
    'spicy': ['pepper', 'cinnamon', 'clove', 'vanilla'],
    'earthy': ['mushroom', 'forest_floor', 'truffle'],
    'oak': ['vanilla', 'toast', 'smoke', 'cedar']
}

# 2. Build graph
graph = build_graph_from_hierarchy(wine_attributes, root='wine')

# 3. Validate (always good practice!)
validation = validate_graph_structure(graph)
print("Validation:", validation['issues'])
print(f"Total attributes: {validation['statistics']['num_descriptions']}")

# 4. Create context
context = create_context_template('wine', 'has', '{0} {1} aroma')

# 5. Compute embeddings
all_attrs = [d for d in graph.descriptions if d != 'wine']
embeddings = compute_embeddings(all_attrs, model_name='BERT', context=context)

# 6. Save results
embeddings_to_csv(embeddings, 'wine_embeddings.csv')
```

### Example 2: Cheese Flavor Wheel

```python
from sensory_cokge import build_graph_from_hierarchy, create_context_template, compute_embeddings

cheese_flavors = {
    'dairy': {
        'fresh': ['milk', 'cream', 'butter', 'yogurt'],
        'aged': ['cultured', 'tangy', 'sour']
    },
    'savory': ['umami', 'salty', 'brothy', 'meaty'],
    'sweet': ['caramel', 'honey', 'nutty'],
    'pungent': {
        'sharp': ['acidic', 'biting'],
        'funky': ['barnyardy', 'ammonia', 'earthy']
    },
    'nutty': ['almond', 'hazelnut', 'walnut', 'cashew']
}

graph = build_graph_from_hierarchy(cheese_flavors, root='cheese')
context = create_context_template('cheese', 'tastes', '{0} {1}')

# Get embeddings for all flavor attributes
flavors = [d for d in graph.descriptions if d != 'cheese']
embeddings = compute_embeddings(flavors, 'BERT', context=context)
```

### Example 3: Coffee Tasting Notes

```python
from sensory_cokge import build_graph_from_csv, create_context_template, compute_embeddings

# If you have coffee_attributes.csv:
# attribute,category
# fruity,root
# nutty,root
# chocolate,root
# berry,fruity
# citrus,fruity
# almond,nutty
# hazelnut,nutty
# dark,chocolate
# milk,chocolate

graph = build_graph_from_csv(
    'coffee_attributes.csv',
    child_column='attribute',
    parent_column='category',
    root='root',
    graph_name='coffee_flavors'
)

context = create_context_template('coffee', 'has', '{0} {1} flavor')
attributes = [d for d in graph.descriptions if d != 'root']
embeddings = compute_embeddings(attributes, 'BERT', context=context)
```

### Example 4: Chocolate Taste Profile

```python
chocolate_profile = {
    'sweet': ['honey', 'caramel', 'vanilla', 'sugar', 'molasses'],
    'bitter': ['cocoa', 'dark', 'burnt', 'roasted'],
    'fruity': {
        'berry': ['raspberry', 'strawberry', 'cherry', 'blueberry'],
        'citrus': ['orange', 'lemon', 'lime'],
        'tropical': ['banana', 'pineapple', 'mango']
    },
    'nutty': ['almond', 'hazelnut', 'walnut', 'pecan'],
    'spicy': ['cinnamon', 'chili', 'ginger', 'cardamom'],
    'earthy': ['tobacco', 'leather', 'wood']
}

graph = build_graph_from_hierarchy(chocolate_profile, root='chocolate')
context = create_context_template('chocolate', 'has', '{0} {1} taste')
```

## Context Template Guidelines

The template follows: **"This [Food] [Verb] [Attribute]."**

### Common Verbs by Sensory Modality

| Sensory Type | Verbs | Example |
|--------------|-------|---------|
| Taste | has, tastes | "This wine tastes {0} {1}" |
| Aroma/Smell | has, smells | "This coffee smells {0} {1}" |
| Texture/Mouthfeel | has, feels | "This cheese feels {0} {1}" |
| Appearance | looks, appears | "This beer looks {0} {1}" |
| General | has | "This chocolate has {0} {1} note" |

### Example Templates

```python
# Wine aroma
create_context_template('wine', 'has', '{0} {1} aroma')
# → "This wine has {0} {1} aroma."

# Cheese texture
create_context_template('cheese', 'feels', '{0} {1}')
# → "This cheese feels {0} {1}."

# Beer appearance
create_context_template('beer', 'has', '{0} {1} appearance')
# → "This beer has {0} {1} appearance."

# Bread flavor
create_context_template('bread', 'has', '{0} {1} flavor')
# → "This bread has {0} {1} flavor."

# Olive oil taste
create_context_template('olive oil', 'tastes', '{0} {1}')
# → "This olive oil tastes {0} {1}."
```

## Model Selection Guide

Choose the best model for your needs:

| Model | Speed | Performance | Best For |
|-------|-------|-------------|----------|
| BERT | Fast | Good | General use, quick experiments |
| RoBERTa | Fast | Better | Production, balanced performance |
| ALBERT | Very Fast | Good | Large datasets, limited GPU |
| BART | Medium | Excellent | Complex attributes |
| T5 | Medium | Excellent | Multi-lingual attributes |
| GPT2 | Fast | Good | Creative/informal descriptions |

```python
# Example: Try different models
models = ['BERT', 'RoBERTa', 'BART']

for model_name in models:
    embeddings = compute_embeddings(
        attributes,
        model_name=model_name,
        context=context
    )
    embeddings_to_csv(embeddings, f'{model_name}_embeddings.csv')
```

## Troubleshooting

### Problem: "Graph contains cycles"

**Solution**: Check your hierarchy for circular references

```python
# Use validation to find the problem
validation = validate_graph_structure(graph)
print(validation['issues'])
```

### Problem: "Some nodes unreachable from root"

**Solution**: Make sure all attributes connect back to root

```python
# Bad: Missing connection
hierarchy = {
    'fruity': ['apple'],
    # 'spicy' is defined but not connected!
}

# Good: All connected
hierarchy = {
    'fruity': ['apple'],
    'spicy': ['cinnamon']  # Connected to root
}
```

### Problem: Column not found in CSV

**Solution**: Check your column names exactly match

```python
# If your CSV has columns: 'flavor', 'group'
graph = build_graph_from_csv(
    'my_file.csv',
    child_column='flavor',  # Match exactly!
    parent_column='group'   # Match exactly!
)
```

## Advanced: Fine-tuning Models

For best results with your specific food:

```python
from sensory_cokge import generate_synthetic_data

# Generate training data from your graph
data = generate_synthetic_data(
    train_samples=50000,
    eval_samples=5000,
    output_dir='./training_data'
)

# Then use the fine-tuning scripts provided
# See finetune_BERT_by_sequence_classification.py for details
```

## Getting Help

1. **Validate your graph first**:
   ```python
   validation = validate_graph_structure(graph)
   print(validation)
   ```

2. **Start simple**: Test with a small attribute set first

3. **Check examples**: Look at the complete workflow examples in this guide

4. **Common issues**:
   - Make sure your CSV has headers
   - Check that all attributes eventually connect to root
   - Avoid duplicate attribute names
   - Use lowercase for consistency

## Summary: Minimal Working Example

```python
from sensory_cokge import (
    build_graph_from_hierarchy,
    create_context_template,
    compute_embeddings,
    embeddings_to_csv
)

# 1. Define your food attributes (nested dictionary)
my_food_attributes = {
    'sweet': ['honey', 'sugar', 'caramel'],
    'sour': ['citrus', 'vinegar', 'acidic'],
    'salty': ['sea_salt', 'cured', 'briny']
}

# 2. Build graph
graph = build_graph_from_hierarchy(
    my_food_attributes,
    root='my_food',
    graph_name='my_food_flavors'
)

# 3. Create context
context = create_context_template('my_food', 'tastes', '{0} {1}')

# 4. Compute embeddings
attrs = [d for d in graph.descriptions if d != 'my_food']
embeddings = compute_embeddings(attrs, 'BERT', context=context)

# 5. Export results
embeddings_to_csv(embeddings, 'my_results.csv')

print("Done! Check my_results.csv")
```

That's it! You now have embeddings for your food's sensory attributes that you can analyze with standard data science tools.
