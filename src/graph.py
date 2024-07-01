import os
import copy
import numpy as np

import coffee_database

__all__ = ['description_graph', 'graph_properties', 'normalize_weighted_adjacency',
        'NOT_COUNT_DESCRIPTIONS', 'CONNENTION_DISTANCES']

NOT_COUNT_DESCRIPTIONS = ['floral (inner layer)', 'floral (middle layer)', 
        'green/vegetative (inner layer)', 'green/vegetative (middle layer)']

CONNENTION_DISTANCES = {
        'forward': {'root-branch': 10.,
                    'sub-category': 1.,
                    'synonym': 0.,
                    'ingredient': 1.,
                    'process': 1.,
                    'unknown': 1.},

        'reverse': {'root-branch': 10.,
                    'sub-category': 1.,
                    'synonym': 0.,
                    'ingredient': 1.,
                    'process': 1.,
                    'unknown': 1.},
}

def description_graph(
        graph_name = 'coffee_flavor_wheel(unduplicated)', 
        exclude_descriptions = NOT_COUNT_DESCRIPTIONS):

    db = coffee_database.CoffeeDatabase()
    graph = coffee_database.description_graph_from_database(db,
            graph_name = graph_name,
            connection_distances = CONNENTION_DISTANCES,
            dynamic = True)

    for description in exclude_descriptions:
        graph.delete_description(description)

    graph.dynamic = False
    return graph

def normalize_weighted_adjacency(A):
    # row-normalized adjacency
    assert len(A.shape) == 2

    min_val = np.min(A)
    max_val = np.max(A)

    norm_A = (A - min_val) / (max_val - min_val)

    return norm_A


def graph_properties(
        graph_name = 'coffee_flavor_wheel(unduplicated)',
        exclude_descriptions = NOT_COUNT_DESCRIPTIONS,
        root_description = False,
        reverse_connection = True,
        saved_files = None):

    properties = {}
    graph = description_graph(graph_name = graph_name,
                              exclude_descriptions = exclude_descriptions)

    (weighted_adjacency,
     descriptions) = graph.weighted_adjacency_matrix(root_description = root_description,
                                                     reverse_connection = reverse_connection)

    properties['descriptions'] = descriptions

    normalized_adjancency = normalize_weighted_adjacency(weighted_adjacency)
    properties['normalized_adjacency'] = normalized_adjancency
    threshold_distances = (CONNENTION_DISTANCES['forward']['root-branch'] +\
            CONNENTION_DISTANCES['forward']['root-branch']) / 2.

    weighted_adjacency = weighted_adjacency * (weighted_adjacency < threshold_distances)
    normalized_adjancency = normalize_weighted_adjacency(weighted_adjacency)
    properties['normalized_adjacency_NoRB'] = normalized_adjancency

    distance_matrix = np.zeros((len(descriptions), len(descriptions)), dtype = np.float32)
    for row_idx in range(len(descriptions)):
        for col_idx in range(len(descriptions)):
            if row_idx != col_idx:
                row_description = descriptions[row_idx]
                col_description = descriptions[col_idx]
                distance_matrix[row_idx, col_idx] = graph.distance_between_descriptions(row_description,
                                                                                        col_description,
                                                                                        weighted = True)
    max_distance = np.max(distance_matrix)
    if max_distance > 0.:
        normalized_distances = distance_matrix / max_distance
    else:
        normalized_distances = copy.deepcopy(distance_matrix)

    properties['normalized_distances'] = normalized_distances
    threshold_distances = (CONNENTION_DISTANCES['forward']['root-branch'] +\
            CONNENTION_DISTANCES['forward']['root-branch'])

    distance_matrix = distance_matrix * (distance_matrix < threshold_distances)
    max_distance = np.max(distance_matrix)
    if max_distance > 0.:
        normalized_distances = distance_matrix / max_distance
    else:
        normalized_distances = copy.deepcopy(distance_matrix)

    properties['normalized_distances_NoRB'] = normalized_distances

    return properties



