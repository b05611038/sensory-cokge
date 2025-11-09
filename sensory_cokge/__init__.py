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
    if graph is None:
        graph_props = graph_properties()
    else:
        graph_props = graph_properties()  # Can be customized with graph parameters

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
    save_csv : bool, optional
        Whether to save data as CSV files. Default: True

    Returns
    -------
    dict
        Dictionary with 'train' and 'eval' keys containing generated data
        Each sample has 'selections' (list of text options) and 'ground_truth' (correct index)

    Examples
    --------
    >>> data = generate_synthetic_data(train_samples=1000, eval_samples=100)
    >>> print(f"Generated {len(data['train'])} training samples")
    >>> print(f"First sample: {data['train'][0]['selections'][0]}")
    """
    data_number = {'train': train_samples, 'eval': eval_samples}

    generated_data = generate_finetune_data(
        data_number=data_number,
        graph_name=graph_name,
        exclude_descriptions=NOT_COUNT_DESCRIPTIONS,
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
