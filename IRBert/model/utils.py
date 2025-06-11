import math

import torch
import torch.nn as nn

__all__ = ['sinusoidal_position_encoding', 'reinit_layer_weight',
        'reduce_model_outputs']

# Reamin some helper function for IRBertProcessor closed.

def sinusoidal_position_encoding(max_seq_length, hidden_size):
    table = torch.zeros((max_seq_length, hidden_size))

    position = torch.arange(0, max_seq_length).unsqueeze(1)
    div_term = (torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size)).exp()

    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)
    table.requries_grad = False

    return table

def reinit_layer_weight(layer, initializer_range):
    assert isinstance(layer, (nn.Linear, nn.Embedding))
    assert isinstance(initializer_range, float)

    nn.init.normal_(layer.weight, mean = 0., std = initializer_range)
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.bias, 0.)

    return None

def reduce_model_outputs(outputs, reduced_outputs):
    assert isinstance(outputs, dict)
    if reduced_outputs is not None:
        assert isinstance(reduced_outputs, (tuple, list))
        for key in outputs:
            if key in reduced_outputs:
                outputs[key] = None

    return outputs


