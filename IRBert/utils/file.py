import json

import torch

from safetensors import safe_open
from safetensors.torch import save_file

__all__ = ['save_as_json', 'load_json', 'save_as_safetensors',
        'load_safetensors', 'save_as_csv']

def save_as_json(obj, filename):
    assert isinstance(filename, str)

    if not filename.endswith('.json'):
        filename += '.json'

    obj = json.dumps(obj)
    with open(filename, 'w') as out_file:
        out_file.write(obj)
        out_file.close()

    return None

def load_json(filename, extension_check = True):
    assert isinstance(filename, str)
    assert isinstance(extension_check, bool)

    content = None
    if extension_check:
        if not filename.endswith('.json'):
            raise RuntimeError('File: {0} is not a .json file.'.format(filename))

    with open(filename, 'r') as in_file:
        content = json.loads(in_file.read())

    return content

def save_as_safetensors(tensors, filename):
    assert isinstance(tensors, dict)
    assert isinstance(filename, str)

    if not filename.endswith('.safetensors'):
        filename += '.safetensors'

    save_file(tensors, filename)
    return None

def load_safetensors(filename, device = 'cpu', extension_check = True):
    assert isinstance(filename, str)
    assert isinstance(device, (str, torch.device))
    assert isinstance(extension_check, bool)

    if extension_check:
        if not filename.endswith('.safetensors'):
            raise RuntimeError('File: {0} is not a .json file.'.format(filename))

    tensors = {}
    with safe_open(filename, framework = 'pt', device = device) as in_files:
        for key in in_files.keys():
            tensors[key] = in_files.get_tensor(key)

    return tensors

def save_as_csv(table, filename, row_names = None, column_names = None):
    assert isinstance(table, torch.Tensor)
    assert len(table.shape) == 2

    if row_names is not None:
        assert isinstance(row_names, (tuple, list))
        assert table.shape[0] == len(row_names)

    if column_names is not None:
        assert isinstance(column_names, (tuple, list))
        assert table.shape[1] == len(column_names)

    assert isinstance(filename, str)
    if not filename.endswith('.csv'):
        filename += '.csv'

    if row_names is not None:
        head_line = ','
    else:
        head_line = ''

    if column_names is not None:
        for col_name in column_names:
            head_line += '{0},'.format(col_name)

        head_line += '\n'
        lines = [head_line]
    else:
        lines = []

    for row_idx in range(table.shape[0]):
        if row_names is not None:
            single_line = '{0},'.format(row_names[row_idx])
        else:
            single_line = ''

        for col_idx in range(table.shape[1]):
            value = float(table[row_idx, col_idx])
            single_line += '{0},'.format(value)

        single_line += '\n'
        lines.append(single_line)

    with open(filename, 'w') as f:
        f.writelines(lines)
        f.close()

    return None




