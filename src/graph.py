import os
import copy
import numpy as np

from igraph import Graph, plot

__all__ = ['description_graph', 'graph_properties', 'normalize_weighted_adjacency',
        'CONNECTIONS', 'NOT_COUNT_DESCRIPTIONS', 'CONNENTION_DISTANCES', 'CoffeeDescriptionGraph']

NOT_COUNT_DESCRIPTIONS = ['floral (inner layer)', 'floral (middle layer)', 
        'green/vegetative (inner layer)', 'green/vegetative (middle layer)']

CONNECTIONS = ['root-branch', 'sub-category', 'synonym', 'unknown']

CONNENTION_DISTANCES = {
        'forward': {'root-branch': 10.,
                    'sub-category': 1.,
                    'synonym': 0.,
                    'unknown': 1.},

        'reverse': {'root-branch': 10.,
                    'sub-category': 1.,
                    'synonym': 0.,
                    'unknown': 1.},
}

class CoffeeDescriptionGraph:
    def __init__(self,
            descriptions,
            connections,
            root = None,
            graph_name = 'SYSTEM', 
            connection_distances = None,
            connection_text_length = 20,
            dynamic = False):

        self.dynamic = dynamic
        self.__init_done = False
        self.__descriptions = []
        self._graph = Graph(directed = True)
        self._undirected_graph = Graph(directed = True)

        self.graph_name = graph_name
        self.connection_distances = connection_distances
        self.connection_text_length = connection_text_length

        assert isinstance(descriptions, (tuple, list))
        for des in descriptions:
            self.add_description(des)

        assert isinstance(connections, (tuple, list))
        for conn in connections:
            self.add_connection(conn)

        self.root = root
        self.__init_done = True

    @property
    def descriptions(self):
        return copy.deepcopy(self.__descriptions)

    @property
    def graph(self):
        return self._graph

    @property
    def undirected_graph(self):
        return self._undirected_graph

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        if root is not None:
            assert isinstance(root, str)
            assert root in self.descriptions

        self._root = root
        return None

    @property
    def graph_name(self):
        return self._graph_name

    @graph_name.setter
    def graph_name(self, graph_name):
        assert isinstance(graph_name, str)
        self._graph_name = graph_name
        return None

    @property
    def connection_distances(self):
        return self._connection_distances
 
    @connection_distances.setter
    def connection_distances(self, connection_distances):
        if connection_distances is not None:
            assert isinstance(connection_distances, dict)
            assert 'forward' in connection_distances.keys()
            assert 'reverse' in connection_distances.keys()

            for direction in ['forward', 'reverse']:
                assert isinstance(connection_distances[direction], dict)
                for conn in CONNECTIONS:
                    assert conn in connection_distances[direction].keys()

        self._connection_distances = connection_distances
        return None

    @property
    def connection_text_length(self):
        return self._connection_text_length

    @connection_text_length.setter
    def connection_text_length(self, connection_text_length):
        assert isinstance(connection_text_length, int)
        assert connection_text_length >= 1
        self._connection_text_length = connection_text_length
        return None

    @property
    def dynamic(self):
        return self._dynamic

    @dynamic.setter
    def dynamic(self, dynamic):
        assert isinstance(dynamic, bool)
        self._dynamic = dynamic
        return None

    def __repr__(self):
        lines = "{0}(graph_name='{1}', dynamic={2})\n"\
                .format(self.__class__.__name__, self.graph_name, self.dynamic)

        return lines

    def _add_description(self, description_name):
        self.__descriptions.append(description_name)
        self.graph.add_vertices(1)
        self.graph.vs[-1]['name'] = description_name
        self.graph.vs[-1]['label'] = description_name
        self.undirected_graph.add_vertices(1)
        self.undirected_graph.vs[-1]['name'] = description_name
        self.undirected_graph.vs[-1]['label'] = description_name
        return None

    def add_description(self, description_name):
        assert isinstance(description_name, str)
        if self.dynamic:
            self._add_description(description_name)
        else:
            if not self.__init_done:
                self._add_description(description_name)
            else:
                print('Because {0}.dynamic=False, cannot add description into graph.'\
                        .format(self.__class__.__name__))

        return None

    def delete_description(self, description_name):
        assert isinstance(description_name, str)
        assert description_name in self.descriptions
        if self.dynamic:
            description_index = self.graph.vs['label'].index(description_name)
            self.graph.delete_vertices(description_index)
            description_index = self.undirected_graph.vs['label'].index(description_name)
            self.undirected_graph.delete_vertices(description_index)
            self.__descriptions.remove(description_name)
        else:
            print('Because {0}.dynamic=False, cannot delete descirption in graph.'\
                    .format(self.__class__.__name__))

        return None

    def _add_connection(self, connection):
        source = connection.get('source', None)
        target = connection.get('target', None)

        path_type = connection.get('path_type', 'unknown')
        forward_weight, reverse_weight = None, None
        if self.connection_distances is not None:
            forward_weight = self.connection_distances['forward'][path_type]
            reverse_weight = self.connection_distances['reverse'][path_type]

        source_idx = self.graph.vs.find(source).index
        target_idx = self.graph.vs.find(target).index
        self.graph.add_edges([(source_idx, target_idx)])
        self.graph.es[-1]['label'] = path_type
        if forward_weight is not None:
            self.graph.es[-1]['weight'] = forward_weight

        source_idx = self.undirected_graph.vs.find(source).index
        target_idx = self.undirected_graph.vs.find(target).index
        self.undirected_graph.add_edges([(source_idx, target_idx)])
        self.undirected_graph.es[-1]['label'] = path_type
        if forward_weight is not None:
            self.undirected_graph.es[-1]['weight'] = forward_weight

        self.undirected_graph.add_edges([(target_idx, source_idx)])
        self.undirected_graph.es[-1]['label'] = '{0} (R)'.format(path_type)
        if reverse_weight is not None:
            self.undirected_graph.es[-1]['weight'] = reverse_weight

        return None

    def add_connection(self, connection):
        assert isinstance(connection, dict)
        for key in ['source', 'target', 'path_type']:
            assert key in connection.keys()

        if self.dynamic:
            self._add_connection(connection)
        else:
            if not self.__init_done:
                self._add_connection(connection)
            else:
                print('Because {0}.dynamic=False, cannot add connection into graph.'\
                        .format(self.__class__.__name__))

    def _delete_connection_in_graph(self, graph_object, description, another_description):
        index = graph_object.vs['label'].index(description)
        another_index = graph_object.vs['label'].index(another_description)
        existing_connection_ids = []
        if graph_object.are_connected(index, another_index):
            connection_id = graph_object.get_eid(index, another_index)
            existing_connection_ids.append(connection_id)

        if graph_object.are_connected(another_index, index):
            connection_id = graph_object.get_eid(another_index, index)
            existing_connection_ids.append(connection_id)

        if len(existing_connection_ids) > 0:
            graph_object.delete_edges(existing_connection_ids)

        return None

    def delete_connection(self, description, another_description):
        description = string_check(description, 'description',
                valid_candidates = self.descriptions)
        another_description = string_check(another_description, 'another_description',
                valid_candidates = self.descriptions)

        if self.dynamic:
            self._delete_connection_in_graph(self.graph, 
                                             description, 
                                             another_description)
            self._delete_connection_in_graph(self.undirected_graph, 
                                             description, 
                                             another_description)
        else:
            print('Because {0}.dynamic=False, cannot delete connection in graph.'\
                    .format(self.__class__.__name__))

        return None

    def get_connection(self, description, another_description, 
                reverse_direction = True, formated_string = True):

        description = string_check(description, 'description',
                 valid_candidates = self.descriptions)
        another_description = string_check(another_description, 'another_description',
                 valid_candidates = self.descriptions)
        reverse_direction = boolean_check(reverse_direction, 'reverse_direction')
        formated_string = boolean_check(formated_string, 'formated_string')

        conn_type = None
        index = self.graph.vs['label'].index(description)
        another_index = self.graph.vs['label'].index(another_description)
        if self.graph.are_connected(index, another_index):
            conn_id = self.graph.get_eid(index, another_index)
            conn_type = self.graph.es['label'][conn_id]
            if formated_string:
                conn_type = '--[{0}]->'.format(conn_type).center(self.connection_text_length)

        elif self.graph.are_connected(another_index, index):
            if reverse_direction:
                conn_id = self.graph.get_eid(another_index, index)
                conn_type = self.graph.es['label'][conn_id]
                if formated_string:
                    conn_type = '<-[{0}]--'.format(conn_type).center(self.connection_text_length)
            else:
                if formated_string:
                    conn_type = 'X'.center(self.connection_text_length) 
        else:
            if formated_string:
                conn_type = 'X'.center(self.connection_text_length)

        if formated_string:
            connection = '{0} {1} {2}'.format(description, conn_type, another_description)
        else:
            connection = conn_type

        return connection

    def valid_construction(self):
        return self.graph.is_dag()

    def parents_of_description(self, description):
        description = string_check(description, 'description',
                 valid_candidates = self.descriptions)

        index = self.graph.vs['label'].index(description)
        parent_indices = self.graph.neighbors(index, mode = 'in')
        parents = [self.graph.vs['label'][idx] for idx in parent_indices]
        return parents

    def children_of_description(self, description):
        description = string_check(description, 'description',
                 valid_candidates = self.descriptions)

        index = self.graph.vs['label'].index(description)
        child_indices = self.graph.neighbors(index, mode = 'out')
        children = [self.graph.vs['label'][idx] for idx in child_indices]
        return children

    def distance_between_descriptions(self, description, another_description, 
            reverse_direction = True, weighted = True):

        assert isinstance(description, str)
        assert description in self.descriptions
        assert isinstance(another_description, str)
        assert another_description in self.descriptions
        assert isinstance(reverse_direction, bool)
        assert isinstance(weighted, bool)

        if reverse_direction:
            index = self.undirected_graph.vs['label'].index(description)
            another_index = self.undirected_graph.vs['label'].index(another_description)
        else:
            index = self.graph.vs['label'].index(description)
            another_index = self.graph.vs['label'].index(another_description)

        connection_weights = None
        if weighted:
            if self.connection_distances is not None:
                if reverse_direction:
                    connection_weights = self.undirected_graph.es['weight']
                else:
                    connection_weights = self.graph.es['weight']

        distance, shortest_distances = float('inf'), []
        if reverse_direction:
            shortest_distances = self.undirected_graph.distances(index, another_index,
                    weights = connection_weights, mode = 'out')
        else:
            shortest_distances = self.graph.distances(index, another_index,
                    weights = connection_weights, mode = 'out')

        for list_obj in shortest_distances:
            for dis in list_obj:
                distance = min(distance, float(dis))

        return distance

    def _get_adjacency_matrix(self, reverse_connection, weighted):
        adjacency_kwargs = {}
        if weighted:
            adjacency_kwargs['attribute'] = 'weight'

        if reverse_connection:
            adjacency_matrix = self.undirected_graph.get_adjacency(**adjacency_kwargs)
            description_ordering = list(self.undirected_graph.vs['label'])
        else:
            adjacency_matrix = self.graph.get_adjacency(**adjacency_kwargs)
            description_ordering = list(self.graph.vs['label'])

        adjacency_matrix = np.array(adjacency_matrix.data, dtype = np.float32)
        return adjacency_matrix, description_ordering        

    def _remove_root_from_matrix(self, adjacency_matrix, description_ordering, reverse):
        if self.root is not None:
            if reverse:
                root_idx = self.undirected_graph.vs['label'].index(self.root)
            else:
                root_idx = self.graph.vs['label'].index(self.root)

            selected_indices = [i for i in range(len(description_ordering))]
            selected_indices.remove(root_idx)

            description_ordering.remove(self.root)
            adjacency_matrix = adjacency_matrix[selected_indices, :][:, selected_indices]

        return adjacency_matrix, description_ordering

    def weighted_adjacency_matrix(self, root_description = False, reverse_connection = False):
        assert isinstance(root_description, bool)
        assert isinstance(reverse_connection, bool)

        (adjacency_matrix,
         description_ordering) = self._get_adjacency_matrix(reverse_connection, True)

        if not root_description:
            (adjacency_matrix,
             description_ordering) = self._remove_root_from_matrix(adjacency_matrix,
                                                                   description_ordering,
                                                                   reverse_connection)

        return adjacency_matrix, description_ordering

    def plot(self, *args, **kwargs):
        return plot(self.graph, *args, **kwargs)


def _recursive_traverse_struc(all_descriptions, parents_of_descriptions, parent_name, struc):
    if isinstance(struc, list):
        for description in struc:
            all_descriptions.append(description)
            if parent_name is not None:
                if description not in parents_of_descriptions.keys():
                    parents_of_descriptions[description] = [parent_name]
                else:
                    parents_of_descriptions[description].append(parent_name)

        return None

    elif isinstance(struc, dict):
        for sub_description in struc:
            all_descriptions.append(sub_description)
            if parent_name is not None:
                if sub_description not in parents_of_descriptions.keys():
                    parents_of_descriptions[sub_description] = [parent_name]
                else:
                    parents_of_descriptions[sub_description].append(parent_name)

            _recursive_traverse_struc(all_descriptions, parents_of_descriptions,
                    sub_description, struc[sub_description])

    return None

def coffee_flavor_wheel_descriptions(duplicate):
    if duplicate:
        layer_structure = {
                'floral (inner layer)': {'floral (middle layer)': ['chamomile', 'rose', 'jasmine'],
                                 'black_tea': []},
                'fruity': {'berry': ['blackberry', 'raspberry', 'blueberry', 'strawberry'],
                           'dried fruit': ['raisin', 'prune'],
                           'other fruit': ['coconut', 'cherry', 'pomegranate', 'pineapple',
                                           'grape', 'apple', 'peach', 'pear'],
                           'citrus fruit': ['graphfruit', 'orange', 'lemon', 'lime']},
                'sour/fermented': {'sour': ['sour aromatics', 'acetic acid', 'butyric acid',
                                            'isovaleric acid', 'citric acid', 'malic acid'],
                                   'alcohol/fermented': ['winey', 'whiskey', 'fermented',
                                                         'overripe']},
                'green/vegetative (inner layer)': {'olive oil': [],
                                             'raw': [],
                                             'green/vegetative (middle layer)': ['under-ripe', 'peapod', 'fresh',
                                                                                 'dark green', 'vegetative',
                                                                                 'hay-like', 'herb-like'],
                                             'beany': []},
                'other': {'papery/musty': ['stale', 'cardboard', 'papery', 'woody', 'moldy/damp',
                                           'musty/dusty', 'musty/earthy', 'animalic', 'meaty/brothy',
                                           'phenolic'],
                          'chemical': ['bitter', 'salty', 'medicinal', 'petroleum', 'skunky', 'rubber']},
                'roasted': {'pipe tobacco': [],
                            'tobacco': [],
                            'burnt': ['acrid', 'ashy', 'smoky', 'brown, roast'],
                            'cereal': ['malt', 'grain']},
                'spices': {'pungent': [],
                           'pepper': [],
                           'brown spice': ['anise', 'nutmeg', 'cinnamon', 'clove']},
                'nutty/cocoa': {'nutty': ['peanuts', 'hazelnut', 'almond'],
                                'cocoa': ['chocolate', 'dark chocolate']},
                'sweet': {'brown sugar': ['molasses', 'mapple syrup', 'caramelized', 'honey'],
                          'vanilla': [],
                          'vanillin': [],
                          'overall sweet': [],
                          'sweet aromatics': []}}
    else:
        layer_structure = {
                'floral': ['black_tea', 'chamomile', 'rose', 'jasmine', 'floral (inner layer)', 
                           'floral (middle layer)'],
                'fruity': {'berry': ['blackberry', 'raspberry', 'blueberry', 'strawberry'],
                           'dried fruit': ['raisin', 'prune'],
                           'other fruit': ['coconut', 'cherry', 'pomegranate', 'pineapple',
                                           'grape', 'apple', 'peach', 'pear'],
                           'citrus fruit': ['graphfruit', 'orange', 'lemon', 'lime']},
                'sour/fermented': {'sour': ['sour aromatics', 'acetic acid', 'butyric acid',
                                            'isovaleric acid', 'citric acid', 'malic acid'],
                                   'alcohol/fermented': ['winey', 'whiskey', 'fermented',
                                                         'overripe']},
                'green/vegetative': ['olive oil', 'raw', 'under-ripe', 'peapod', 'fresh', 'dark green', 
                                     'vegetative', 'hay-like', 'herb-like', 'beany', 'green/vegetative (inner layer)', 
                                     'green/vegetative (middle layer)'],
                'other': {'papery/musty': ['stale', 'cardboard', 'papery', 'woody', 'moldy/damp',
                                           'musty/dusty', 'musty/earthy', 'animalic', 'meaty/brothy',
                                           'phenolic'],
                          'chemical': ['bitter', 'salty', 'medicinal', 'petroleum', 'skunky', 'rubber']},
                'roasted': {'pipe tobacco': [],
                            'tobacco': [],
                            'burnt': ['acrid', 'ashy', 'smoky', 'brown, roast'],
                            'cereal': ['malt', 'grain']},
                'spices': {'pungent': [],
                           'pepper': [],
                           'brown spice': ['anise', 'nutmeg', 'cinnamon', 'clove']},
                'nutty/cocoa': {'nutty': ['peanuts', 'hazelnut', 'almond'],
                                'cocoa': ['chocolate', 'dark chocolate']},
                'sweet': {'brown sugar': ['molasses', 'mapple syrup', 'caramelized', 'honey'],
                          'vanilla': [],
                          'vanillin': [],
                          'overall sweet': [],
                          'sweet aromatics': []}}


    all_descriptions, parents_of_descriptions = ['root'], {}
    for inner_description in layer_structure:
        parents_of_descriptions[inner_description] = ['root']

    _recursive_traverse_struc(all_descriptions, parents_of_descriptions, None, layer_structure)
    all_descriptions = list(set(all_descriptions))

    return all_descriptions, parents_of_descriptions

def connections_transform_from_parents_of_descriptions(parents_of_descriptions, duplicate, root_name = 'root'):
    connections = []
    for des in parents_of_descriptions:
        parents = parents_of_descriptions[des]
        for parent in parents:
            source = parent
            target = des
            path_type = 'sub-category'
            if not duplicate:
                if ('(inner layer)' in target) or ('(middle layer)' in target):
                    path_type = 'synonym'

            if root_name in source:
                path_type = 'root-branch'

            single_connection = {'source': source,
                                 'target': target,
                                 'path_type': path_type}

            connections.append(single_connection)

    return connections

def description_graph(
        graph_name = 'coffee_flavor_wheel(unduplicated)', 
        connection_distances = CONNENTION_DISTANCES,
        exclude_descriptions = NOT_COUNT_DESCRIPTIONS):

    assert isinstance(graph_name, str)
    assert graph_name in ['coffee_flavor_wheel(unduplicated)', 'coffee_flavor_wheel']
    if graph_name == 'coffee_flavor_wheel':
        duplicate = True
    else:
        duplicate = False

    (all_descriptions,
     parents_of_descriptions) = coffee_flavor_wheel_descriptions(duplicate = duplicate)

    connections = connections_transform_from_parents_of_descriptions(parents_of_descriptions,
                                                                     duplicate)

    graph = CoffeeDescriptionGraph(all_descriptions,
                                   connections,
                                   root = 'root',
                                   graph_name = graph_name,
                                   connection_distances = connection_distances,
                                   dynamic = True)

    if not duplicate:
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



