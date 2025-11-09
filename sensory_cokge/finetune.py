import random
import hashlib

from .utils import use_a_or_an
from .graph import description_graph, NOT_COUNT_DESCRIPTIONS

__all__ = ['generate_finetune_data']

def _simple_hash(sampled_list):
    list_string = ','.join(map(str, sampled_list))
    return hashlib.sha256(list_string.encode()).hexdigest()

def _construct_single_data(descriptions, graph, food_name='coffee'):
    codes = [['one', 'two', 'three'], ['A', 'B', 'C'], ['alpha', 'beta', 'gamma']]
    context = '{0} {{0}} has {{1}} {{2}} flavors; '.format(food_name.capitalize())

    code_names = random.sample(codes, 1)[0]
    overlapping_text = ''
    for des_idx in range(len(descriptions)):
        description = descriptions[des_idx]
        overlapping_text += context.format(code_names[des_idx], use_a_or_an(description), description)

    distances = []
    for i in range(len(descriptions)):
       for j in range(i + 1, len(descriptions)):
           distances.append(graph.distance_between_descriptions(descriptions[i], descriptions[j]))

    selections, ground_truth = [], -1
    for i in range(len(descriptions)):
       for j in range(i + 1, len(descriptions)):
           non_include_idx = [k for k in range(len(descriptions))]
           non_include_idx.remove(i)
           non_include_idx.remove(j)
           non_include_idx = non_include_idx[0]

           compare_text = '{{0}} {0} and {{1}} {0} taste more similar, while {{2}} {0} tastes different.'.format(food_name)
           compare_text = compare_text.format(code_names[i], code_names[j], code_names[non_include_idx])
           selections.append(overlapping_text + compare_text)

           if graph.distance_between_descriptions(descriptions[i], descriptions[j]) <= min(distances):
               ground_truth = len(selections) -1

    similar_text = '{{0}} {0}, {{1}} {0}, and {{2}} {0} all taste very similar.'.format(food_name)
    similar_text = similar_text.format(*tuple(code_names))
    selections.append(overlapping_text + similar_text)
    if max(distances) <= 2.:
        ground_truth = len(selections) -1

    different_text = '{{0}} {0}, {{1}} {0}, and {{2}} {0} all taste very different.'.format(food_name)
    different_text = different_text.format(*tuple(code_names))
    selections.append(overlapping_text + different_text)
    if min(distances) >= 20.:
        ground_truth = len(selections) -1

    single_data = {'selections': selections,
                   'ground_truth': ground_truth}

    return single_data

def generate_finetune_data(
        data_number = {'train': 50000, 'eval': 10000},
        graph_name = 'coffee_flavor_wheel(unduplicated)',
        graph = None,
        food_name = None,
        exclude_descriptions = NOT_COUNT_DESCRIPTIONS,
        progress = True,
        display_split = 100):

    # Handle graph creation - either use provided graph or create from graph_name
    if graph is None:
        graph = description_graph(graph_name = graph_name,
                                  exclude_descriptions = exclude_descriptions)
        # If no food_name provided and using default graph, use 'coffee'
        if food_name is None:
            food_name = 'coffee'
    else:
        # Using custom graph - extract graph_name if available, or use 'food' as default
        if food_name is None:
            # Try to infer food name from graph_name attribute, or use root, or default to 'food'
            if hasattr(graph, 'graph_name'):
                # Extract food name from graph_name (e.g., 'wine_flavors' -> 'wine')
                food_name = graph.graph_name.split('_')[0]
            elif hasattr(graph, 'root') and graph.root is not None:
                food_name = graph.root
            else:
                food_name = 'food'

    descriptions = list(graph.descriptions)
    descriptions.remove(graph.root)
    description_index_list = [i for i in range(len(descriptions))]

    generated_data = {}
    already_sampled_combinations = []
    total_sampled_number = 0
    for set_name in data_number:
        generated_data[set_name] = []
        total_sampled_number += data_number[set_name]

    all_generated_data = []
    sample_number = 0
    while sample_number < total_sampled_number:
        sampled_indices = random.sample(description_index_list, 3)
        hashed_indices = _simple_hash(sampled_indices)
        if hashed_indices in already_sampled_combinations:
            continue

        already_sampled_combinations.append(hashed_indices)
        sampled_descriptions = [descriptions[i] for i in sampled_indices]
        single_data = _construct_single_data(sampled_descriptions, graph, food_name=food_name)
        all_generated_data.append(single_data)
        sample_number += 1

        if progress:
            if sample_number % display_split == 0:
                print('Generation progress: {0} / {1}'.format(sample_number,
                                                              total_sampled_number))

    start_index, end_index = 0, 0
    for set_name in data_number:
        set_number = data_number[set_name]
        end_index += set_number

        for idx in range(start_index, end_index):
            generated_data[set_name].append(all_generated_data[idx])

        start_index = end_index

    return generated_data


