"""
sensory_cokge: A toolkit for sensory description graph analysis and LLM embeddings

This package provides tools to:
1. Transform text descriptions to embeddings (exportable as CSV)
2. Evaluate embedding structural consistency with description graphs
3. Generate synthetic training data
4. Fine-tune language models for sensory analysis
"""

import os
import pandas as pd
import torch
import numpy as np

# Core modules
from .graph import (
    description_graph,
    graph_properties,
    DescriptionGraph,
    normalize_weighted_adjacency,
    CONNECTIONS,
    NOT_COUNT_DESCRIPTIONS,
    CONNENTION_DISTANCES
)

from .models import (
    ALBERT_NAME, ALBERT_embeddings,
    BERT_NAME, BERT_embeddings,
    BART_NAME, BART_embeddings,
    GEMMA2_NAME, Gemma2_embeddings,
    GPT2_NAME, GPT2_embeddings,
    LLAMA3_NAME, Llama3_embeddings,
    QWEN2_NAME, Qwen2_embeddings,
    RoBERTa_NAME, RoBERTa_embeddings,
    T5_NAME, T5_embeddings,
    Laser_embeddings
)

from .finetune import generate_finetune_data

from .metrics import EvaluationMetrics

from .utils import (
    init_directory,
    save_object,
    load_pickle_obj,
    load_json_obj,
    parse_training_args,
    use_a_or_an
)

from .relative_embedding import (
    construct_relative_embeddings,
    RelativeEmbedding,
    DescriptorColors,
    AnchorDescriptors
)

from .distances import (
    cosine_similarity,
    angle_differences,
    l2_differences
)


__version__ = '0.1.0'

__all__ = [
    # Core helper functions
    'compute_embeddings',
    'embeddings_to_csv',
    'evaluate_embeddings',
    'generate_synthetic_data',
    'create_description_graph',

    # Food researcher helpers
    'build_graph_from_hierarchy',
    'build_graph_from_csv',
    'create_context_template',
    'validate_graph_structure',

    # Graph functions
    'description_graph',
    'graph_properties',
    'DescriptionGraph',
    'normalize_weighted_adjacency',

    # Model embeddings
    'ALBERT_embeddings', 'ALBERT_NAME',
    'BERT_embeddings', 'BERT_NAME',
    'BART_embeddings', 'BART_NAME',
    'GEMMA2_embeddings', 'GEMMA2_NAME',
    'GPT2_embeddings', 'GPT2_NAME',
    'Llama3_embeddings', 'LLAMA3_NAME',
    'Qwen2_embeddings', 'QWEN2_NAME',
    'RoBERTa_embeddings', 'RoBERTa_NAME',
    'T5_embeddings', 'T5_NAME',
    'Laser_embeddings',

    # Synthetic data generation
    'generate_finetune_data',

    # Evaluation
    'EvaluationMetrics',

    # Utilities
    'init_directory',
    'save_object',
    'load_pickle_obj',
    'load_json_obj',
    'parse_training_args',
    'use_a_or_an',

    # Relative embeddings
    'construct_relative_embeddings',
    'RelativeEmbedding',
    'DescriptorColors',
    'AnchorDescriptors',

    # Distance measures
    'cosine_similarity',
    'angle_differences',
    'l2_differences',

    # Constants
    'CONNECTIONS',
    'NOT_COUNT_DESCRIPTIONS',
    'CONNENTION_DISTANCES',
]


# ============================================================================
# Core Helper Functions
# ============================================================================

def compute_embeddings(
        descriptions,
        model_name='BERT',
        context=None,
        batch_size=4,
        device='auto',
        pretrained_model=None,
        finetuned_model=None):
    """
    Transform text descriptions to embeddings using specified language model.

    This is the core function for converting sensory descriptions into
    vector representations that can be analyzed and compared.

    Parameters
    ----------
    descriptions : list of str
        List of sensory descriptions to embed
    model_name : str, optional
        Model architecture to use. Options: 'ALBERT', 'BERT', 'BART', 'GPT2',
        'RoBERTa', 'T5', 'Gemma2', 'Llama3', 'Qwen2', 'Laser'
        Default: 'BERT'
    context : str, optional
        Context template with placeholders for article and description.
        Example: "This coffee has {0} {1} flavor."
        If None, uses default context from models module
    batch_size : int, optional
        Batch size for inference. Default: 4
    device : str, optional
        Computing device ('auto', 'cuda', 'cpu'). Default: 'auto'
    pretrained_model : str, optional
        Path or name of pretrained model. If None, uses default
    finetuned_model : str, optional
        Path to finetuned model. If provided, overrides pretrained_model

    Returns
    -------
    dict
        Dictionary mapping indices to embedding information:
        {idx: {'description': str, 'text': str, 'encoder_embedding': Tensor, ...}}

    Examples
    --------
    >>> descriptions = ['fruity', 'floral', 'nutty']
    >>> embeddings = compute_embeddings(descriptions, model_name='BERT')
    >>> for idx in embeddings:
    ...     print(embeddings[idx]['description'], embeddings[idx]['encoder_embedding'].shape)
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model_map = {
        'ALBERT': (ALBERT_embeddings, ALBERT_NAME),
        'BERT': (BERT_embeddings, BERT_NAME),
        'BART': (BART_embeddings, BART_NAME),
        'GPT2': (GPT2_embeddings, GPT2_NAME),
        'RoBERTa': (RoBERTa_embeddings, RoBERTa_NAME),
        'T5': (T5_embeddings, T5_NAME),
        'Gemma2': (Gemma2_embeddings, GEMMA2_NAME),
        'Llama3': (Llama3_embeddings, LLAMA3_NAME),
        'Qwen2': (Qwen2_embeddings, QWEN2_NAME),
        'Laser': (Laser_embeddings, None),
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")

    embed_func, default_model = model_map[model_name]

    kwargs = {
        'descriptions': descriptions,
        'batch_size': batch_size,
        'device': device,
    }

    if context is not None:
        kwargs['context'] = context

    if finetuned_model is not None:
        kwargs['finetuned_model'] = finetuned_model
    elif pretrained_model is not None:
        kwargs['pretrained_model'] = pretrained_model
    elif default_model is not None:
        kwargs['pretrained_model'] = default_model

    return embed_func(**kwargs)


def embeddings_to_csv(embeddings, filepath, embedding_type='encoder_embedding'):
    """
    Export embeddings to CSV file for tabular analysis.

    Converts embedding dictionary to a CSV file where each row represents
    a description and its embedding vector components.

    Parameters
    ----------
    embeddings : dict
        Embedding dictionary from compute_embeddings()
    filepath : str
        Output CSV file path
    embedding_type : str, optional
        Type of embedding to export ('encoder_embedding', 'decoder_embedding',
        'relative_encoder_embedding', 'relative_decoder_embedding')
        Default: 'encoder_embedding'

    Returns
    -------
    str
        Path to saved CSV file

    Examples
    --------
    >>> embeddings = compute_embeddings(['fruity', 'floral'], model_name='BERT')
    >>> embeddings_to_csv(embeddings, 'embeddings.csv')
    'embeddings.csv'
    """
    if not filepath.endswith('.csv'):
        filepath += '.csv'

    data = []
    for idx in embeddings:
        row = {'description': embeddings[idx]['description']}

        if embedding_type in embeddings[idx]:
            embedding_vec = embeddings[idx][embedding_type]
            if isinstance(embedding_vec, torch.Tensor):
                embedding_vec = embedding_vec.cpu().numpy()

            # Add each dimension as a column
            for i, val in enumerate(embedding_vec):
                row[f'dim_{i}'] = float(val)

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

    print(f"Embeddings exported to {filepath}")
    print(f"Shape: {len(data)} descriptions x {len(df.columns)-1} dimensions")

    return filepath


def evaluate_embeddings(embeddings, graph=None, embedding_type='encoder_embedding'):
    """
    Evaluate structural consistency of embeddings with description graph.

    Compares the distance relationships in the embedding space with the
    graph structure to assess how well the model captures semantic relationships.

    Parameters
    ----------
    embeddings : dict
        Embedding dictionary from compute_embeddings()
    graph : DescriptionGraph, optional
        Description graph for comparison. If None, uses default coffee flavor wheel
    embedding_type : str, optional
        Type of embedding to evaluate. Default: 'encoder_embedding'

    Returns
    -------
    dict
        Evaluation metrics including:
        - 'adjacency_matching_l2': L2-based adjacency matching score
        - 'adjacency_matching_angle': Angle-based adjacency matching score
        - 'distances_matching_l2': L2-based distance matching score
        - 'distances_matching_angle': Angle-based distance matching score
        (with and without root-branch connections: *_NoRB variants)

    Examples
    --------
    >>> embeddings = compute_embeddings(['fruity', 'floral'], model_name='BERT')
    >>> results = evaluate_embeddings(embeddings)
    >>> print(f"L2 matching: {results['distances_matching_l2']:.4f}")
    """
    import copy

    if graph is None:
        # Use default coffee flavor wheel
        graph_props = graph_properties()
    else:
        # Extract properties from the custom graph
        # This follows the same logic as graph_properties() in graph.py
        graph_props = {}

        # Get weighted adjacency matrix (with reverse connections)
        root_description = False
        reverse_connection = True
        (weighted_adjacency,
         descriptions) = graph.weighted_adjacency_matrix(root_description=root_description,
                                                         reverse_connection=reverse_connection)

        graph_props['descriptions'] = descriptions

        # Normalize adjacency matrix
        normalized_adjacency = normalize_weighted_adjacency(weighted_adjacency)
        graph_props['normalized_adjacency'] = normalized_adjacency

        # Create version without root-branch connections
        threshold_distances = (CONNENTION_DISTANCES['forward']['root-branch'] +
                              CONNENTION_DISTANCES['forward']['root-branch']) / 2.
        weighted_adjacency_NoRB = weighted_adjacency * (weighted_adjacency < threshold_distances)
        normalized_adjacency_NoRB = normalize_weighted_adjacency(weighted_adjacency_NoRB)
        graph_props['normalized_adjacency_NoRB'] = normalized_adjacency_NoRB

        # Compute distance matrix between all descriptions
        distance_matrix = np.zeros((len(descriptions), len(descriptions)), dtype=np.float32)
        for row_idx in range(len(descriptions)):
            for col_idx in range(len(descriptions)):
                if row_idx != col_idx:
                    row_description = descriptions[row_idx]
                    col_description = descriptions[col_idx]
                    distance_matrix[row_idx, col_idx] = graph.distance_between_descriptions(
                        row_description,
                        col_description,
                        weighted=True
                    )

        # Normalize distances
        max_distance = np.max(distance_matrix)
        if max_distance > 0.:
            normalized_distances = distance_matrix / max_distance
        else:
            normalized_distances = copy.deepcopy(distance_matrix)

        graph_props['normalized_distances'] = normalized_distances

        # Create version without root-branch connections
        threshold_distances = (CONNENTION_DISTANCES['forward']['root-branch'] +
                              CONNENTION_DISTANCES['forward']['root-branch'])
        distance_matrix_NoRB = distance_matrix * (distance_matrix < threshold_distances)
        max_distance = np.max(distance_matrix_NoRB)
        if max_distance > 0.:
            normalized_distances_NoRB = distance_matrix_NoRB / max_distance
        else:
            normalized_distances_NoRB = copy.deepcopy(distance_matrix_NoRB)

        graph_props['normalized_distances_NoRB'] = normalized_distances_NoRB

    # Determine source structure from embedding type
    if 'decoder' in embedding_type:
        source_struc = 'decoder'
    else:
        source_struc = 'encoder'

    evaluator = EvaluationMetrics(
        model_name='custom',
        source_struc=source_struc,
        embedding_properties=embeddings,
        graph_properties=graph_props
    )

    results = {
        'adjacency_matching_l2': evaluator.adjacency_matching_l2,
        'adjacency_matching_l2_NoRB': evaluator.adjacency_matching_l2_NoRB,
        'distances_matching_l2': evaluator.distances_matching_l2,
        'distances_matching_l2_NoRB': evaluator.distances_matching_l2_NoRB,
        'adjacency_matching_angle': evaluator.adjacency_matching_angle,
        'adjacency_matching_angle_NoRB': evaluator.adjacency_matching_angle_NoRB,
        'distances_matching_angle': evaluator.distances_matching_angle,
        'distances_matching_angle_NoRB': evaluator.distances_matching_angle_NoRB,
    }

    return results


def generate_synthetic_data(
        train_samples=50000,
        eval_samples=10000,
        output_dir='./outputs',
        graph_name='coffee_flavor_wheel(unduplicated)',
        graph=None,
        food_name=None,
        save_csv=True):
    """
    Generate synthetic training data for fine-tuning language models.

    Creates comparison tasks based on the description graph structure,
    where models learn to identify which descriptions are most similar.

    Parameters
    ----------
    train_samples : int, optional
        Number of training samples to generate. Default: 50000
    eval_samples : int, optional
        Number of evaluation samples to generate. Default: 10000
    output_dir : str, optional
        Directory to save CSV files. Default: './outputs'
    graph_name : str, optional
        Name of description graph to use. Default: 'coffee_flavor_wheel(unduplicated)'
        Only used if graph=None
    graph : DescriptionGraph, optional
        Custom description graph for your food product. If provided, overrides graph_name.
        Default: None (uses coffee flavor wheel)
    food_name : str, optional
        Name of the food product for generated text (e.g., 'wine', 'cheese', 'chocolate').
        If None, will be inferred from graph.graph_name or graph.root.
        Default: None (uses 'coffee' for default graph)
    save_csv : bool, optional
        Whether to save data as CSV files. Default: True

    Returns
    -------
    dict
        Dictionary with 'train' and 'eval' keys containing generated data
        Each sample has 'selections' (list of text options) and 'ground_truth' (correct index)

    Examples
    --------
    >>> # Default: Generate data for coffee flavor wheel
    >>> data = generate_synthetic_data(train_samples=1000, eval_samples=100)
    >>> print(f"Generated {len(data['train'])} training samples")
    >>> print(f"First sample: {data['train'][0]['selections'][0]}")

    >>> # Custom: Generate data for wine
    >>> wine_graph = build_graph_from_hierarchy({'fruity': ['apple', 'pear']}, root='wine')
    >>> wine_data = generate_synthetic_data(
    ...     train_samples=1000,
    ...     eval_samples=100,
    ...     graph=wine_graph,
    ...     food_name='wine',
    ...     output_dir='./wine_data'
    ... )
    """
    data_number = {'train': train_samples, 'eval': eval_samples}

    generated_data = generate_finetune_data(
        data_number=data_number,
        graph_name=graph_name,
        graph=graph,
        food_name=food_name,
        exclude_descriptions=NOT_COUNT_DESCRIPTIONS if graph is None else [],
        progress=True
    )

    if save_csv:
        init_directory(output_dir)

        for set_name in generated_data:
            filepath = os.path.join(output_dir, f'{set_name}.csv')

            # Layout data as CSV
            assert len(generated_data[set_name]) > 0
            number_of_selection = len(generated_data[set_name][0]['selections'])

            header = ''
            for idx in range(number_of_selection):
                header += f'text{idx},'
            header += 'ground_truth\n'

            lines = [header]
            for single_data in generated_data[set_name]:
                single_line = ''
                for single_text in single_data['selections']:
                    single_line += '"' + single_text + '",'
                single_line += f"{single_data['ground_truth']}\n"
                lines.append(single_line)

            with open(filepath, 'w') as f:
                f.writelines(lines)

            print(f'Saved {set_name} data to {filepath}')

    return generated_data


def create_description_graph(
        descriptions,
        connections,
        root=None,
        graph_name='custom',
        connection_distances=None):
    """
    Create a custom description graph for your domain.

    Build a hierarchical graph of sensory descriptions with typed connections
    (e.g., 'sub-category', 'synonym', 'root-branch').

    Parameters
    ----------
    descriptions : list of str
        All description nodes in the graph
    connections : list of dict
        Connections between nodes. Each dict should have:
        {'source': str, 'target': str, 'path_type': str}
        where path_type is one of: 'root-branch', 'sub-category', 'synonym', 'unknown'
    root : str, optional
        Root node of the graph. Default: None
    graph_name : str, optional
        Name for the graph. Default: 'custom'
    connection_distances : dict, optional
        Distance weights for each connection type. Default: uses CONNENTION_DISTANCES

    Returns
    -------
    DescriptionGraph
        Graph object with methods for querying distances and relationships

    Examples
    --------
    >>> descriptions = ['root', 'fruit', 'apple', 'banana']
    >>> connections = [
    ...     {'source': 'root', 'target': 'fruit', 'path_type': 'root-branch'},
    ...     {'source': 'fruit', 'target': 'apple', 'path_type': 'sub-category'},
    ...     {'source': 'fruit', 'target': 'banana', 'path_type': 'sub-category'}
    ... ]
    >>> graph = create_description_graph(descriptions, connections, root='root')
    >>> print(graph.distance_between_descriptions('apple', 'banana'))
    """
    if connection_distances is None:
        connection_distances = CONNENTION_DISTANCES

    graph = DescriptionGraph(
        descriptions=descriptions,
        connections=connections,
        root=root,
        graph_name=graph_name,
        connection_distances=connection_distances,
        dynamic=False
    )

    return graph


# ============================================================================
# Helper Functions for Food Researchers
# ============================================================================

def build_graph_from_hierarchy(
        hierarchy,
        root='root',
        graph_name='custom_food_graph',
        connection_distances=None):
    """
    Build a description graph from a simple nested dictionary hierarchy.

    This is the easiest way to define a sensory attribute graph for your food
    product. Simply provide a nested dictionary where keys are parent categories
    and values are either lists (leaf nodes) or dicts (sub-categories).

    HOW TO DEFINE A VALID GRAPH (DAG - Directed Acyclic Graph)
    ===========================================================

    Rules for creating a valid sensory description graph:

    1. **Hierarchical Structure**: The graph must be tree-like with a single root
       - Start with one root node (e.g., 'root', 'wine', 'cheese')
       - Branch out to main categories
       - Continue to sub-categories and specific attributes

    2. **No Cycles**: Avoid circular references
       - ✓ Valid: root → fruity → apple
       - ✗ Invalid: root → fruity → apple → fruity (cycle!)

    3. **Connection Types** (automatically assigned):
       - 'root-branch': Direct children of root node
       - 'sub-category': Parent-child relationships in the tree
       - 'synonym': For duplicate/alternative names (use carefully)

    4. **Format Options**:
       - List: ['attr1', 'attr2'] → Creates leaf nodes
       - Dict: {'category': [...]} → Creates sub-category with children
       - Empty list: [] → Creates a node without children

    Parameters
    ----------
    hierarchy : dict
        Nested dictionary defining the attribute hierarchy.
        Format: {category: [attributes]} or {category: {subcategory: [...]}}
    root : str, optional
        Name of the root node. Default: 'root'
    graph_name : str, optional
        Descriptive name for your graph. Default: 'custom_food_graph'
    connection_distances : dict, optional
        Custom distance weights for connection types.
        Default: Uses standard distances (root-branch=10, sub-category=1, synonym=0)

    Returns
    -------
    DescriptionGraph
        Graph object ready for embedding analysis

    Examples
    --------
    **Example 1: Simple wine aroma graph**

    >>> wine_hierarchy = {
    ...     'fruity': ['apple', 'pear', 'citrus', 'berry'],
    ...     'floral': ['rose', 'violet', 'jasmine'],
    ...     'spicy': ['pepper', 'cinnamon', 'clove'],
    ...     'earthy': ['mushroom', 'soil', 'forest']
    ... }
    >>> graph = build_graph_from_hierarchy(wine_hierarchy, root='wine',
    ...                                     graph_name='wine_aromas')

    **Example 2: Nested cheese flavor graph**

    >>> cheese_hierarchy = {
    ...     'dairy': {
    ...         'fresh': ['milk', 'cream', 'butter'],
    ...         'cultured': ['yogurt', 'sour_cream']
    ...     },
    ...     'savory': ['umami', 'salty', 'broth'],
    ...     'pungent': {
    ...         'aged': ['sharp', 'tangy'],
    ...         'fermented': ['funky', 'ammonia']
    ...     }
    ... }
    >>> graph = build_graph_from_hierarchy(cheese_hierarchy, root='cheese',
    ...                                     graph_name='cheese_flavors')

    **Example 3: Chocolate taste profile**

    >>> chocolate_hierarchy = {
    ...     'sweet': ['sugar', 'honey', 'caramel'],
    ...     'bitter': ['cocoa', 'dark', 'roasted'],
    ...     'fruity': {
    ...         'berry': ['raspberry', 'cherry'],
    ...         'citrus': ['orange', 'lemon']
    ...     },
    ...     'nutty': ['almond', 'hazelnut'],
    ...     'spicy': ['cinnamon', 'chili']
    ... }
    >>> graph = build_graph_from_hierarchy(chocolate_hierarchy, root='chocolate')
    >>> # Now you can use it!
    >>> from sensory_cokge import compute_embeddings, evaluate_embeddings
    >>> descriptions = ['raspberry', 'cherry', 'almond', 'honey']
    >>> embeddings = compute_embeddings(descriptions, model_name='BERT',
    ...                                  context='This chocolate has {0} {1} note.')
    """
    # Collect all descriptions and parent relationships
    all_descriptions = [root]
    parents_of_descriptions = {}

    def _traverse_hierarchy(parent_name, structure):
        """Recursively traverse the hierarchy to extract nodes and edges."""
        if isinstance(structure, list):
            # Leaf nodes
            for item in structure:
                if item not in all_descriptions:
                    all_descriptions.append(item)
                if item not in parents_of_descriptions:
                    parents_of_descriptions[item] = [parent_name]
                else:
                    parents_of_descriptions[item].append(parent_name)

        elif isinstance(structure, dict):
            # Sub-categories
            for sub_category, sub_structure in structure.items():
                if sub_category not in all_descriptions:
                    all_descriptions.append(sub_category)
                if sub_category not in parents_of_descriptions:
                    parents_of_descriptions[sub_category] = [parent_name]
                else:
                    parents_of_descriptions[sub_category].append(parent_name)

                # Recursively process children
                _traverse_hierarchy(sub_category, sub_structure)

    # Build the parent relationships
    for main_category, structure in hierarchy.items():
        if main_category not in all_descriptions:
            all_descriptions.append(main_category)
        parents_of_descriptions[main_category] = [root]
        _traverse_hierarchy(main_category, structure)

    # Convert to connections format
    connections = []
    for description, parents in parents_of_descriptions.items():
        for parent in parents:
            # Determine connection type
            if parent == root:
                path_type = 'root-branch'
            else:
                path_type = 'sub-category'

            connections.append({
                'source': parent,
                'target': description,
                'path_type': path_type
            })

    # Create the graph
    return create_description_graph(
        descriptions=all_descriptions,
        connections=connections,
        root=root,
        graph_name=graph_name,
        connection_distances=connection_distances
    )


def build_graph_from_csv(
        filepath,
        child_column='attribute',
        parent_column='category',
        root='root',
        graph_name='custom_graph'):
    """
    Build a description graph from a CSV file defining parent-child relationships.

    This function is useful when you have your sensory attributes organized
    in a spreadsheet. Simply export to CSV with columns for attributes and
    their parent categories.

    CSV Format Requirements
    =======================
    - Must have at least 2 columns: one for attributes, one for parents
    - Parent of root should be empty/blank or equal to root itself
    - Each row defines one attribute and its parent category

    Parameters
    ----------
    filepath : str
        Path to CSV file
    child_column : str, optional
        Name of column containing attribute names. Default: 'attribute'
    parent_column : str, optional
        Name of column containing parent categories. Default: 'category'
    root : str, optional
        Name of the root node. Default: 'root'
    graph_name : str, optional
        Descriptive name for the graph. Default: 'custom_graph'

    Returns
    -------
    DescriptionGraph
        Graph object ready for embedding analysis

    Examples
    --------
    **CSV file format (beer_flavors.csv):**
    ```
    attribute,category
    malty,root
    hoppy,root
    fruity,root
    caramel,malty
    toasted,malty
    citrus,hoppy
    pine,hoppy
    tropical,fruity
    berry,fruity
    ```

    >>> graph = build_graph_from_csv('beer_flavors.csv',
    ...                               child_column='attribute',
    ...                               parent_column='category',
    ...                               root='root',
    ...                               graph_name='beer_flavors')
    >>> print(f"Graph has {len(graph.descriptions)} attributes")
    """
    # Read CSV
    df = pd.read_csv(filepath)

    # Validate columns
    if child_column not in df.columns:
        raise ValueError(f"Column '{child_column}' not found in CSV. "
                        f"Available columns: {list(df.columns)}")
    if parent_column not in df.columns:
        raise ValueError(f"Column '{parent_column}' not found in CSV. "
                        f"Available columns: {list(df.columns)}")

    # Collect all unique descriptions
    all_descriptions = set([root])
    parents_of_descriptions = {}

    for _, row in df.iterrows():
        child = str(row[child_column]).strip()
        parent = str(row[parent_column]).strip()

        # Handle empty parent (treat as root)
        if not parent or parent == 'nan' or parent.lower() == 'none':
            parent = root

        all_descriptions.add(child)
        all_descriptions.add(parent)

        if child not in parents_of_descriptions:
            parents_of_descriptions[child] = []
        if parent not in parents_of_descriptions[child]:
            parents_of_descriptions[child].append(parent)

    # Convert to connections
    connections = []
    for child, parents in parents_of_descriptions.items():
        for parent in parents:
            # Determine connection type
            if parent == root:
                path_type = 'root-branch'
            else:
                path_type = 'sub-category'

            connections.append({
                'source': parent,
                'target': child,
                'path_type': path_type
            })

    return create_description_graph(
        descriptions=list(all_descriptions),
        connections=connections,
        root=root,
        graph_name=graph_name
    )


def create_context_template(food_name, verb, attribute_placeholder='{0} {1}'):
    """
    Create a context template for embedding generation following the pattern:
    "This [Food] [Verb] [Attribute]."

    This helper makes it easy to generate proper context strings for different
    foods without worrying about grammar or formatting.

    The pattern follows the research paper format where:
    - {0} is the article (a/an) - automatically determined
    - {1} is the sensory attribute

    Parameters
    ----------
    food_name : str
        Name of the food product (e.g., 'wine', 'cheese', 'chocolate', 'coffee')
    verb : str
        Verb describing the sensory experience. Common options:
        - 'has' (general): "This wine has {0} {1} flavor"
        - 'tastes' (taste): "This cheese tastes {0} {1}"
        - 'smells' (aroma): "This coffee smells {0} {1}"
        - 'feels' (texture): "This bread feels {0} {1}"
    attribute_placeholder : str, optional
        Template for attribute insertion. Default: '{0} {1}'
        {0} = article (a/an), {1} = attribute

    Returns
    -------
    str
        Context template string ready to use with compute_embeddings()

    Examples
    --------
    >>> # Wine tasting notes
    >>> wine_context = create_context_template('wine', 'has', '{0} {1} flavor')
    >>> print(wine_context)
    'This wine has {0} {1} flavor.'

    >>> # Cheese flavor
    >>> cheese_context = create_context_template('cheese', 'tastes', '{0} {1}')
    >>> print(cheese_context)
    'This cheese tastes {0} {1}.'

    >>> # Coffee aroma
    >>> coffee_context = create_context_template('coffee', 'smells', '{0} {1}')
    >>> print(coffee_context)
    'This coffee smells {0} {1}.'

    >>> # Bread texture
    >>> bread_context = create_context_template('bread', 'feels', '{0} {1}')
    >>> print(bread_context)
    'This bread feels {0} {1}.'

    >>> # Use with compute_embeddings
    >>> from sensory_cokge import compute_embeddings
    >>> beer_context = create_context_template('beer', 'has', '{0} {1} taste')
    >>> descriptions = ['hoppy', 'malty', 'fruity']
    >>> embeddings = compute_embeddings(descriptions, model_name='BERT',
    ...                                  context=beer_context)

    >>> # More complex examples
    >>> # For tea: "This tea has {0} {1} aroma"
    >>> tea_context = create_context_template('tea', 'has', '{0} {1} aroma')

    >>> # For honey: "This honey tastes {0} {1}"
    >>> honey_context = create_context_template('honey', 'tastes', '{0} {1}')

    >>> # For olive oil: "This olive oil has {0} {1} note"
    >>> oil_context = create_context_template('olive oil', 'has', '{0} {1} note')
    """
    # Ensure food_name doesn't have leading "This"
    food_name = food_name.strip()
    if food_name.lower().startswith('this '):
        food_name = food_name[5:]

    # Build the template
    template = f"This {food_name} {verb} {attribute_placeholder}."

    return template


def validate_graph_structure(graph):
    """
    Validate that a description graph is properly structured as a DAG.

    Checks for common issues:
    1. Graph is a valid DAG (Directed Acyclic Graph - no cycles)
    2. Root is properly set
    3. All nodes are reachable from root
    4. No orphaned nodes

    Parameters
    ----------
    graph : DescriptionGraph
        The graph to validate

    Returns
    -------
    dict
        Validation results with keys:
        - 'is_valid': bool - Overall validity
        - 'is_dag': bool - Whether it's a proper DAG
        - 'has_root': bool - Whether root is set
        - 'all_reachable': bool - All nodes reachable from root
        - 'issues': list of str - Description of any problems found
        - 'statistics': dict - Graph statistics

    Examples
    --------
    >>> hierarchy = {
    ...     'fruity': ['apple', 'pear'],
    ...     'spicy': ['pepper', 'cinnamon']
    ... }
    >>> graph = build_graph_from_hierarchy(hierarchy, root='food')
    >>> results = validate_graph_structure(graph)
    >>> if results['is_valid']:
    ...     print("Graph is valid!")
    ... else:
    ...     print("Issues found:", results['issues'])

    >>> # Check statistics
    >>> print(f"Total attributes: {results['statistics']['num_descriptions']}")
    >>> print(f"Total connections: {results['statistics']['num_connections']}")
    """
    issues = []
    is_valid = True

    # Check if it's a DAG
    is_dag = graph.valid_construction()
    if not is_dag:
        issues.append("Graph contains cycles - must be a Directed Acyclic Graph (DAG)")
        is_valid = False

    # Check if root is set
    has_root = graph.root is not None
    if not has_root:
        issues.append("No root node set - use root parameter when creating graph")
        is_valid = False

    # Check if all nodes are reachable from root
    all_reachable = True
    unreachable = []
    if has_root:
        for desc in graph.descriptions:
            if desc != graph.root:
                distance = graph.distance_between_descriptions(graph.root, desc,
                                                               reverse_direction=True,
                                                               weighted=False)
                if distance == float('inf'):
                    all_reachable = False
                    unreachable.append(desc)

        if not all_reachable:
            issues.append(f"Some nodes unreachable from root: {unreachable[:5]}"
                         + (f" and {len(unreachable)-5} more" if len(unreachable) > 5 else ""))
            is_valid = False

    # Collect statistics
    num_descriptions = len(graph.descriptions)
    num_connections = len(graph.graph.es)

    # Count connection types
    connection_types = {}
    for edge in graph.graph.es:
        edge_type = edge['label']
        connection_types[edge_type] = connection_types.get(edge_type, 0) + 1

    statistics = {
        'num_descriptions': num_descriptions,
        'num_connections': num_connections,
        'connection_types': connection_types,
        'has_root': has_root,
        'root_name': graph.root if has_root else None
    }

    if is_valid:
        issues.append("Graph structure is valid!")

    return {
        'is_valid': is_valid,
        'is_dag': is_dag,
        'has_root': has_root,
        'all_reachable': all_reachable,
        'issues': issues,
        'statistics': statistics
    }


# ============================================================================
# Complete Workflow Example for Food Researchers
# ============================================================================

# Uncomment and run this example to see the complete workflow:
"""
# Example: Complete workflow for analyzing chocolate taste descriptors

# Step 1: Define your attribute graph (easy nested dictionary)
chocolate_attributes = {
    'sweet': ['honey', 'caramel', 'vanilla', 'sugar'],
    'bitter': ['cocoa', 'dark', 'burnt'],
    'fruity': {
        'berry': ['raspberry', 'strawberry', 'cherry'],
        'citrus': ['orange', 'lemon']
    },
    'nutty': ['almond', 'hazelnut', 'walnut'],
    'spicy': ['cinnamon', 'chili', 'ginger']
}

# Step 2: Build the graph
from sensory_cokge import build_graph_from_hierarchy, validate_graph_structure

graph = build_graph_from_hierarchy(
    chocolate_attributes,
    root='chocolate',
    graph_name='chocolate_taste_wheel'
)

# Step 3: Validate your graph
validation = validate_graph_structure(graph)
print("Graph validation:", validation)

# Step 4: Create appropriate context for your food
from sensory_cokge import create_context_template

context = create_context_template('chocolate', 'has', '{0} {1} taste')
print("Context template:", context)

# Step 5: Compute embeddings for your attributes
from sensory_cokge import compute_embeddings

all_attributes = [desc for desc in graph.descriptions if desc != 'chocolate']
embeddings = compute_embeddings(
    all_attributes,
    model_name='BERT',
    context=context,
    device='auto'
)

# Step 6: Export to CSV for analysis
from sensory_cokge import embeddings_to_csv

embeddings_to_csv(embeddings, 'chocolate_embeddings.csv')

# Step 7: Evaluate how well embeddings match your graph structure
from sensory_cokge import evaluate_embeddings

results = evaluate_embeddings(embeddings, graph=graph)
print("Evaluation results:", results)

# Step 8: Generate synthetic data for fine-tuning (optional)
from sensory_cokge import generate_synthetic_data

synthetic_data = generate_synthetic_data(
    train_samples=10000,
    eval_samples=1000,
    output_dir='./chocolate_training_data'
)
"""
