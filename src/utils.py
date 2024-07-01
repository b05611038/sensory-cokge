import os
import json
import pickle

__all__ = ['init_directory', 'save_object', 'load_pickle_obj', 'load_json_obj',
        'use_a_or_an']

def init_directory(path):
    assert isinstance(path, str)

    if not os.path.isdir(path):
        os.makedirs(path)

    return path

def save_object(fname, obj, mode = 'pickle', extension = None):
    if not isinstance(fname, str):
        raise TypeError(fname, "is not string object, it can't be the saving name of file.")

    if mode == 'pickle':
        if extension is None:
            if not fname.endswith('.pkl'):
                fname += '.pkl'

        else:
            if not isinstance(extension, str):
                raise TypeError('File extension must be a str object.')

            if not fname.endswith('.' + extension):
                fname = fname + '.' + extension

        with open(fname, 'wb') as out_file:
            pickle.dump(obj, out_file)

    elif mode == 'json':
        if not fname.endswith('.json'):
            fname += '.json'

        obj = json.dumps(obj)
        with open(fname, 'w') as out_file:
            out_file.write(obj) 

    out_file.close()

    return None

def load_pickle_obj(fname, extension_check = True):
    if extension_check:
        if not fname.endswith('.pkl'):
            raise RuntimeError(fname, 'is not a pickle file.')

    with open(fname, 'rb') as in_file:
        return pickle.load(in_file)

    return None

def load_json_obj(fname):
    if not fname.endswith('.json'):
        raise RuntimeError(fname, 'is not a json file.')

    with open(fname, 'r') as in_file:
        return json.loads(in_file.read())

    return None

def use_a_or_an(word):
    vowels = 'aeiou'
    special_cases = ["honest", "hour", "heir", "honor"]

    word_lower = word.lower()

    if word_lower in special_cases:
        return "an"

    if word_lower[0] in vowels:
        return "an"
    else:
        return "a"


