import torch
import torch.nn as nn

from .utils import (sinusoidal_position_encoding,
                    reinit_layer_weight)


__all__ = ['PositionalEncoding', 'ProcessorConv1dStack', 'TransformerEncoderLayer',
        'TransformerEncoder', 'MultiTaskPredictor']


class PositionalEncoding(nn.Module):
    def __init__(self,
            method = 'absolute', 
            hidden_size = 768, 
            max_seq_length = 4096): 

        super(PositionalEncoding, self).__init__()

        assert method in ['absolute', 'rotary', 'learnable']
        if method == 'rotary' or method == 'learnable':
            raise NotImplementedError()
        else:
            self.fix_weight = True

        self.method = method
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length

        if method == 'absolute':
            self.fixed_positional_encoding = sinusoidal_position_encoding(max_seq_length,
                                                                          hidden_size)
            self.learnable_positional_encoding = None
        else:
            self.fixed_positional_encoding = None
            self.learnable_positional_encoding = None

    @property
    def fix_weight(self):
        return self._fix_weight

    @fix_weight.setter
    def fix_weight(self, fix_weight):
        assert isinstance(fix_weight, bool)
        self._fix_weight = fix_weight
        return None

    def forward(self, embeddings):
        seq_length = embeddings.shape[1]
        dtype, device = embeddings.dtype, embeddings.device
        if self.fixed_positional_encoding.device != device:
            self.fixed_positional_encoding = self.fixed_positional_encoding.to(device)

        if self.fix_weight:
            positional_encoding = self.fixed_positional_encoding[: seq_length].clone().detach()

        positional_encoding = positional_encoding.to(dtype)

        return embeddings + positional_encoding


class ProcessorConv1dStack(nn.Module):
    """
    Placeholder for IR-BERT's spectral CNN processor.

    NOTE: Actual convolutional stack is withheld in the public demo repository
    to protect architectural innovation related to spectral embedding.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("IR-BERT's processor CNN is not included in the demo version.")

    def forward(self, x):
        raise NotImplementedError("IR-BERT's processor CNN is not included in the demo version.")


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,
            hidden_size = 768,
            num_hidden_head = 12,
            feedforward_size = 3072,
            layer_norm_eps = 1e-12,
            dropout_prob = 0.1,
            hidden_act = 'gelu',
            bias = True,
            initializer_range = 2e-3):

        super(TransformerEncoderLayer, self).__init__(
                d_model = hidden_size,
                nhead = num_hidden_head,
                dim_feedforward = feedforward_size,
                dropout = dropout_prob,
                activation = hidden_act,
                layer_norm_eps = layer_norm_eps,
                batch_first = True,
                bias = bias)

        self.hidden_size = hidden_size
        self.num_hidden_head = num_hidden_head
        self.feedforward_size = feedforward_size
        self.dropout_prob = dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias

        reinit_layer_weight(self.linear1, initializer_range)
        reinit_layer_weight(self.linear2, initializer_range)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        (attn_output, 
         attn_weights) = self.self_attn(src, src, src, 
                                        attn_mask = src_mask, 
                                        key_padding_mask = src_key_padding_mask)

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, 
             hidden_size = 768, 
             num_hidden_layer = 12, 
             num_hidden_head = 12,
             feedforward_size = 3072, 
             layer_norm_eps = 1e-12,
             hidden_act = 'gelu', 
             dropout_prob = 0.1,
             initializer_range = 2e-3):

        super(TransformerEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layer = num_hidden_layer
        self.num_hidden_head = num_hidden_head
        self.feedforward_size = feedforward_size
        self.hidden_act = hidden_act
        self.dropout_prob = dropout_prob

        layers = []
        for idx in range(num_hidden_layer):
            layers.append(TransformerEncoderLayer(hidden_size = hidden_size,
                                                  num_hidden_head = num_hidden_head,
                                                  feedforward_size = feedforward_size,
                                                  layer_norm_eps = layer_norm_eps,
                                                  dropout_prob = dropout_prob,
                                                  hidden_act = hidden_act,
                                                  initializer_range = initializer_range))

        self.layers = nn.ModuleList(layers)

    def forward(self, src, src_mask = None, src_key_padding_mask = None, only_output_last = False):
        assert isinstance(only_output_last, bool)

        all_outputs, all_attn_weights = [], []
        for layer in self.layers:
            src, attn_weights = layer(src, src_mask = src_mask,
                                           src_key_padding_mask = src_key_padding_mask)

            if not only_output_last:
                all_outputs.append(src)
                all_attn_weights.append(attn_weights)

        return src, attn_weights, all_outputs, all_attn_weights


class MultiTaskPredictor(nn.Module):
    def __init__(self, 
            task_tokens = (),
            tokens_head_size = {}, 
            hidden_size = 768,
            dropout_prob = 0.1):

        super(MultiTaskPredictor, self).__init__()

        assert isinstance(task_tokens, (tuple, list))
        assert isinstance(tokens_head_size, dict)
        assert isinstance(hidden_size, int)
        assert isinstance(dropout_prob, float)

        for token in task_tokens:
            assert len(token) > 0    
            assert isinstance(tokens_head_size[token], int)

        self.task_tokens = list(task_tokens)
        self.tokens_head_size = tokens_head_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        head_linear = {}
        for token in task_tokens:
             output_size = tokens_head_size[token]
             head_linear[token] = nn.Sequential(nn.Dropout(dropout_prob),
                                                nn.Linear(hidden_size, output_size))

        self.head_linear = nn.ModuleDict(head_linear)

    def add_task_token(self, token, output_size):
        assert isinstance(token, str)
        assert token not in self.head_linear.keys()

        assert isinstance(output_size, int)
        assert output_size > 0

        self.task_tokens.append(token)
        self.tokens_head_size[token] = output_size
        self.head_linear[token] = nn.Sequential(nn.Dropout(self.dropout_prob),
                                                nn.Linear(self.hidden_size, output_size))

        return None

    def delete_task_token(self, token):
        assert isinstance(token, str)
        assert token in self.head_linear.keys()

        self.task_tokens.remove(token)
        del self.tokens_head_size[token]
        module = self.head_linear.pop(token)
        return module

    def forward(self, 
            output_embeddings, 
            task_tokens, 
            task_tokens_indices):

        assert output_embeddings.dim() == 3
        assert isinstance(task_tokens, (tuple, list))
        assert isinstance(task_tokens_indices, dict)

        for token in task_tokens:
            assert token in self.head_linear.keys()
            assert isinstance(task_tokens_indices[token], (torch.LongTensor, torch.cuda.LongTensor))

        predictor_output = []
        for token in task_tokens:
            selected_embeddings = output_embeddings[:, task_tokens_indices[token], :]
            head_output = self.head_linear[token](selected_embeddings)
            predictor_output.append(head_output)

        return torch.stack(predictor_output).permute(1, 0, 2)


