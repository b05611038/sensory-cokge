import torch
import numpy as np

from .graph import (normalize_weighted_adjacency,
                    CONNENTION_DISTANCES)

from .distances import (angle_differences, 
                        l2_differences)

__all__ = ['EvaluationMetrics']


class EvaluationMetrics:
    def __init__(self, model_name, source_struc, embedding_properties, graph_properties):
        self.model_name = model_name
        assert source_struc in ['encoder', 'decoder']
        self.source_struc = source_struc

        self.embedding_properties = embedding_properties
        self.graph_properties = graph_properties

        (self._adjacency_matching_l2,
         self._adjacency_matching_l2_NoRB,
         self._distances_matching_l2,
         self._distances_matching_l2_NoRB) = self._evaluate(embedding_properties,
                                                            graph_properties,
                                                            l2_differences)

        (self._adjacency_matching_angle,
         self._adjacency_matching_angle_NoRB,
         self._distances_matching_angle,
         self._distances_matching_angle_NoRB) = self._evaluate(embedding_properties,
                                                               graph_properties,
                                                               angle_differences)

    def _find_description_index_in_embeddings(self, description, embeddings):
        detect_index = None
        for idx in embeddings:
            if embeddings[idx]['description'] == description:
                detect_index = idx
                break

        return detect_index

    def _evaluate(self, embedding_properties, graph_properties, distance_measure):
        adjacency_matching, adjacency_matching_NoRB = None, None
        distances_matching, distances_matching_NoRB = None, None
        descriptions = graph_properties.get('descriptions', [])
        if len(descriptions) > 0:
            embedding_distances = np.zeros((len(descriptions), len(descriptions)), dtype = np.float32)
            for row_idx in range(len(descriptions)):
                for col_idx in range(len(descriptions)):
                    row_des_idx = self._find_description_index_in_embeddings(descriptions[row_idx],
                                                                             embedding_properties)
                    col_des_idx = self._find_description_index_in_embeddings(descriptions[col_idx],
                                                                             embedding_properties)
                    if row_des_idx is None or col_des_idx is None:
                        print('When dealing descriptions: ({0}, {1}) cannot find their index'\
                                .format(descriptions[row_idx], descriptions[col_idx]))
                        continue

                    key_name = '{0}_embedding'.format(self.source_struc)
                    row_embedding = embedding_properties[row_des_idx][key_name]
                    col_embedding = embedding_properties[col_des_idx][key_name]

                    distance = distance_measure(row_embedding, col_embedding)
                    embedding_distances[row_idx, col_idx] = distance

            threshold = 1e-8
            normalized_adjacency = graph_properties.get('normalized_adjacency', None)
            if normalized_adjacency is not None:
                valid_indices = (normalized_adjacency > threshold)
                embedding_adjacency = embedding_distances * valid_indices
                embedding_adjacency = normalize_weighted_adjacency(embedding_adjacency) 
                adjacency_diff = normalized_adjacency - embedding_adjacency
                adjacency_matching = np.power(adjacency_diff, 2).sum() / valid_indices.sum()

            normalized_adjacency_NoRB = graph_properties.get('normalized_adjacency_NoRB', None)
            if normalized_adjacency_NoRB is not None:
                valid_indices = (normalized_adjacency_NoRB > threshold)
                embedding_adjacency = embedding_distances * valid_indices
                embedding_adjacency = normalize_weighted_adjacency(embedding_adjacency)
                adjacency_diff = normalized_adjacency_NoRB - embedding_adjacency
                adjacency_matching_NoRB = np.power(adjacency_diff, 2).sum() / valid_indices.sum()

            normalized_distances = graph_properties.get('normalized_distances', None)
            if normalized_distances is not None:
                valid_indices = (normalized_distances > threshold)
                embedding_dists = embedding_distances * valid_indices
                max_distance = np.max(embedding_dists)
                if max_distance > 0.:
                    embedding_dists /= max_distance

                distances_diff = normalized_distances - embedding_dists
                distances_matching = np.power(distances_diff, 2).sum() / valid_indices.sum()

            normalized_distances_NoRB = graph_properties.get('normalized_distances_NoRB', None)
            if normalized_distances_NoRB is not None:
                valid_indices = (normalized_distances_NoRB > threshold)
                embedding_dists = embedding_distances * valid_indices
                max_distance = np.max(embedding_dists)
                if max_distance > 0.:
                    embedding_dists /= max_distance

                distances_diff = normalized_distances_NoRB - embedding_dists
                distances_matching_NoRB = np.power(distances_diff, 2).sum() / valid_indices.sum()

        return (adjacency_matching, 
                adjacency_matching_NoRB, 
                distances_matching, 
                distances_matching_NoRB)

    @property
    def adjacency_matching_l2(self):
        return self._adjacency_matching_l2

    @property
    def adjacency_matching_l2_NoRB(self):
        return self._adjacency_matching_l2_NoRB

    @property
    def adjacency_matching_angle(self):
        return self._adjacency_matching_angle

    @property
    def adjacency_matching_angle_NoRB(self):
        return self._adjacency_matching_angle_NoRB
 
    @property
    def distances_matching_l2(self):
        return self._distances_matching_l2

    @property
    def distances_matching_l2_NoRB(self):
        return self._distances_matching_l2_NoRB

    @property
    def distances_matching_angle(self):
        return self._distances_matching_angle

    @property
    def distances_matching_angle_NoRB(self):
        return self._distances_matching_angle_NoRB

    @property
    def metrics(self):
        metric_dict = {
                'adjacency_matching_l2': self.adjacency_matching_l2,
                'adjacency_matching_l2_NoRB': self.adjacency_matching_l2_NoRB,
                'adjacency_matching_angle': self.adjacency_matching_angle,
                'adjacency_matching_angle_NoRB': self.adjacency_matching_angle_NoRB,
                'distances_matching_l2': self.distances_matching_l2,
                'distances_matching_l2_NoRB': self.distances_matching_l2_NoRB,
                'distances_matching_angle': self.distances_matching_angle,
                'distances_matching_angle_NoRB': self.distances_matching_angle_NoRB,
        }

        return metric_dict

    def __repr__(self):
        return "{0}(model_name='{1}', source_struc='{2}')".format(self.__class__.__name__, 
                self.model_name, self.source_struc)

    def summary(self):
        text = self.__repr__() + ':'
        max_text_length = len(text)
        metric_dict = self.metrics
        print(text)
        for m in metric_dict:
            single_line = '{0}: {1:.5f}'.format(m, metric_dict[m])
            max_text_length = max(max_text_length, len(single_line))
            print(single_line)

        print('end'.center(max_text_length, '-'))
        return None


