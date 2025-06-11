import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (reinit_layer_weight,
                    reduce_model_outputs)

from .base import BaseModel
from .config import IRBertConfig

from .module import (TransformerEncoder,
                     MultiTaskPredictor)

from .tokenizer import (IRBertTokenizer,
                        MultiTaskTokenizer)

from .processor import IRBertProcessor

from .modeling_output import (IRBertModelOutput,
                              IRBertForPreTrainingOutput,
                              IRBertForMaskedSMOutput,
                              IRBertForMultiTaskPredictionOutput)

from .module import (TransformerEncoder,
                     PositionalEncoding)

from ..utils import (init_directory,
                     save_as_safetensors,
                     load_safetensors)


__all__ = ['IRBertModel', 'IRBertForPreTraining', 'IRBertForMaskedSM', 'IRBertForMultiTaskPrediction']


class IRBertModel(BaseModel):
    def __init__(self, config):
        super(IRBertModel, self).__init__()
        assert isinstance(config, IRBertConfig)

        self.config = config

        mask_token_id = config.get('mask_token_id', 0)
        pad_token_id = config.get('pad_token_id', 1)
        sep_token_id = config.get('sep_token_id', 2)

        vocab_size = config.get('vocab_size', 3)
        hidden_size = config.get('hidden_size', 768)
        num_hidden_layer = config.get('num_hidden_layer', 12)
        num_hidden_head = config.get('num_hidden_head', 12)
        feedforward_size = config.get('feedforward_size', 3072)
        hidden_act = config.get('hidden_act', 'gelu')
        dropout_prob = config.get('dropout_prob', 0.1)
        layer_norm_eps = config.get('layer_norm_eps', 1e-12)
        attn_mask_value = config.get('attn_mask_value', -1e9)

        max_position_embeddings = config.get('max_position_embeddings', 4096)
        position_embedding_type = config.get('position_embedding_type', 'absolute')
        initializer_range = config.get('initializer_range', 0.02)

        assert isinstance(mask_token_id, int)
        assert isinstance(pad_token_id, int)
        assert isinstance(sep_token_id, int)

        assert isinstance(max_position_embeddings, int)
        assert max_position_embeddings > 0

        assert isinstance(position_embedding_type, str)
        assert position_embedding_type in ['absolute', 'rotary', 'learnable']

        assert isinstance(hidden_size, int)
        assert isinstance(num_hidden_layer, int)
        assert isinstance(num_hidden_head, int)
        assert isinstance(feedforward_size, int)
        assert isinstance(hidden_act, str)
        assert isinstance(dropout_prob, float)
        assert isinstance(layer_norm_eps, float)

        assert isinstance(initializer_range, float)
        self.initializer_range = initializer_range

        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type

        self.hidden_size = hidden_size
        self.num_hidden_layer = num_hidden_layer
        self.num_hidden_head = num_hidden_head
        self.feedforward_size = feedforward_size
        self.hidden_act = hidden_act
        self.dropout_prob = dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.attn_mask_value = attn_mask_value

        self.positional_embedding = PositionalEncoding(method = position_embedding_type,
                                                       hidden_size = hidden_size,
                                                       max_seq_length = max_position_embeddings)

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        reinit_layer_weight(self.token_embedding, initializer_range)

        self.encoder = TransformerEncoder(hidden_size = hidden_size,
                                          num_hidden_layer = num_hidden_layer,
                                          num_hidden_head = num_hidden_head,
                                          feedforward_size = feedforward_size,
                                          hidden_act = hidden_act,
                                          dropout_prob = dropout_prob,
                                          layer_norm_eps = layer_norm_eps,
                                          initializer_range = initializer_range)

    def modify_token_embedding(self, token_number, replaced_token_indices = None):
        assert isinstance(token_number, int)
        device = self.token_embedding.weight.data.device
        dtype = self.token_embedding.weight.data.dtype
        
        if replaced_token_indices is None:
            replaced_token_indices = torch.arange(self.token_embedding.num_embeddings, device = device)
            replaced_token_indices.max() < self.token_embedding.num_embeddings

        assert isinstance(replaced_token_indices, (torch.LongTensor, torch.cuda.LongTensor))
        assert replaced_token_indices.dim() == 1

        embedding_size = self.hidden_size
        replaced_embeddings = self.token_embedding.weight.data
        embeddings = torch.randn((token_number, embedding_size), dtype = dtype, device = device)
        embeddings = embeddings.normal_(std = self.initializer_range)
        embeddings[: len(replaced_token_indices)] = replaced_embeddings[replaced_token_indices]

        self.token_embedding = nn.Embedding.from_pretrained(embeddings)
        self.config.vocab_size = token_number
        self.config.update('vocab_size', token_number)

        return None

    def forward(self, forward(self, *args, **kwargs):
        raise NotImplementedError(
            "IR-BERT's spectral embedding and fusion logic are not included in this public demo."
        )

        return IRBertModelOutput(**outputs)

    def from_pretrained(self, 
            path, 
            config_filename = 'config.json', 
            filename = 'model.safetensors'):

        assert isinstance(filename, str)
        assert len(filename) > 0
        if not filename.endswith('.safetensors'):
            filename += '.safetensors'

        self = super(IRBertModel, self).from_pretrained(path,
                                                        config_filename = config_filename,
                                                        tokenizer_filename = tokenizer_filename)

        model_filename = os.path.join(path, filename)
        state_dict = load_safetensors(model_filename)
        self.load_state_dict(state_dict)

        return self

    def save_pretrained(self, 
            path, 
            config_filename = 'config.json',
            filename = 'model.safetensors'):

        assert isinstance(path, str)
        assert isinstance(config_filename, str)
        assert len(config_filename) > 0
        if not config_filename.endswith('.json'):
            config_filename += '.json'

        path = init_directory(path)
        config_filename = os.path.join(path, config_filename)
        self.config.to_json_file(config_filename)

        assert isinstance(filename, str)
        assert len(filename) > 0
        if not filename.endswith('.safetensors'):
            filename += '.safetensors'

        assert isinstance(path, str)
        if not os.path.isdir(path):
            init_directory(path)

        cpu_model = self.cpu()
        state_dict = cpu_model.state_dict()

        model_filename = os.path.join(path, filename)
        save_as_safetensors(state_dict, model_filename)

        return None


class IRBertForPreTraining(BaseModel):
    def __init__(self, 
            config, 
            tokenizer = None,
            processor = None,
            use_cache = True):

        super(IRBertForPreTraining, self).__init__()

        assert isinstance(config, IRBertConfig)
        self.config = config
        if tokenizer is not None:
            assert isinstance(tokenizer, IRBertTokenizer)
        else:
            tokenizer = IRBertTokenizer(config = config)

        self.tokenizer = tokenizer

        # ⛔ Processor logic is redacted
        # If needed, load a mock processor
        if processor is not None:
            assert isinstance(processor, IRBertProcessor)
        else:
            self.processor = None  # Processor logic withheld for demo

        # NOTE: The IRBertProcessor module, which handles spectrum–wavelength tokenization,
        # is excluded from this public demo repository to protect unpublished innovation.

        self.processor = processor
        self.model = IRBertModel(config)

        self.use_cache = use_cache

    @property
    def use_cache(self):
        return self.processor.use_cache

    @use_cache.setter
    def use_cache(self, use_cache):
        assert isinstance(use_cache, bool)
        self.processor.use_cache = use_cache
        return None

    @property
    def mask_token_id(self):
        return self.model.mask_token_id

    @property
    def pad_token_id(self):
        return self.model.pad_token_id

    @property
    def sep_token_id(self):
        return self.model.sep_token_id

    @property
    def max_position_embeddings(self):
        return self.model.max_position_embeddings

    @property
    def position_embedding_type(self):
        return self.model.position_embedding_type 

    @property
    def initializer_range(self):
        return self.model.initializer_range

    @property
    def hidden_size(self):
        return self.model.hidden_size

    @property
    def num_hidden_layer(self):
        return self.model.num_hidden_layer

    @property
    def num_hidden_head(self):
        return self.model.num_hidden_head

    @property
    def feedforward_size(self):
        return self.model.feedforward_size

    @property
    def hidden_act(self):
        return self.model.hidden_act

    @property
    def dropout_prob(self):
        return self.model.dropout_prob

    @property
    def layer_norm_eps(self):
        return self.model.layer_norm_eps

    @property
    def attn_mask_value(self):
        return self.model.attn_mask_value

    def add_tokens(self, tokens):
        self.tokenizer.add_tokens(tokens)
        self.model.modify_token_embedding(self.tokenizer.token_number)
        return None

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "IR-BERT's spectral embedding and fusion logic are not included in this public demo."
        )

        return IRBertForPreTrainingOutput(**outputs) 

    def from_pretrained(cls, 
            path, 
            config_filename = 'config.json', 
            tokenizer_filename = 'vocab.json',
            filename = 'model.safetensors'):

        assert isinstance(filename, str)
        assert len(filename) > 0
        if not filename.endswith('.safetensors'):
            filename += '.safetensors'

        self = super(IRBertForPreTraining, cls).from_pretrained(path,
                                                                config_filename = config_filename)

        self.tokenizer = self.tokenizer.from_pretrained(path,
                                                        filename = tokenizer_filename)

        parameter_names = []
        for name, _ in self.named_parameters():
            parameter_names.append(name)
                     
        model_filename = os.path.join(path, filename)
        state_dict = load_safetensors(model_filename)

        filtered_state_dict = {k: v for k, v in state_dict.items() if k in parameter_names}
        self.load_state_dict(filtered_state_dict, strict = False)

        return self

    def save_pretrained(self,
            path, 
            config_filename = 'config.json',
            tokenizer_filename = 'vocab.json',
            processor_filename = 'processor.safetensors',
            filename = 'model.safetensors'):

        assert isinstance(path, str)
        assert isinstance(config_filename, str)
        assert len(config_filename) > 0
        if not config_filename.endswith('.json'):
            config_filename += '.json'

        path = init_directory(path)

        self.config.vocab_size = self.tokenizer.token_number
        self.config.update('vocab_size', self.tokenizer.token_number)

        config_filename = os.path.join(path, config_filename)
        self.config.to_json_file(config_filename)

        self.tokenizer.save_pretrained(path,
                                       filename = tokenizer_filename)

        assert isinstance(filename, str)
        assert len(filename) > 0
        if not filename.endswith('.safetensors'):
            filename += '.safetensors'

        self.processor.save_pretrained(path,
                                       filename = processor_filename)

        cpu_model = self.cpu()
        state_dict = cpu_model.state_dict()

        model_filename = os.path.join(path, filename)
        save_as_safetensors(state_dict, model_filename)

        return None


class IRBertForMaskedSM(IRBertForPreTraining):
    def __init__(self,
            config,
            tokenizer = None,
            processor = None,
            use_cache = True):

        super(IRBertForMaskedSM, self).__init__(
                config = config,
                tokenizer = tokenizer,
                processor = processor,
                use_cache = use_cache)

        self.reconstruction_dropout = nn.Dropout(self.dropout_prob)
        self.reconstruction_predictor = nn.Linear(self.hidden_size, 1)
        reinit_layer_weight(self.reconstruction_predictor, self.initializer_range)

    def forward(self,
            spectrum,
            spectrum_wavelengths,
            concat_tokenized_texts = None,
            spectrum_mask = None,
            wavelengths_mask = None,
            reduced_outputs = None,
            only_output_last = False):

        wave_number = spectrum_wavelengths.shape[-1]
        outputs = super(IRBertForMaskedSM, self).forward(spectrum = spectrum,
                                                         spectrum_wavelengths = spectrum_wavelengths,
                                                         concat_tokenized_texts = concat_tokenized_texts,
                                                         spectrum_mask = spectrum_mask,
                                                         wavelengths_mask = wavelengths_mask,
                                                         only_output_last = only_output_last)

        outputs = dict(outputs)
        last_hidden_state = outputs['last_hidden_state']
        reconstructed_spectrum = self.reconstruction_dropout(last_hidden_state[:, : wave_number])
        reconstructed_spectrum = self.reconstruction_predictor(reconstructed_spectrum)
        reconstructed_spectrum = reconstructed_spectrum.squeeze(-1)
        if spectrum_mask is not None:
            reconstructed_masked_spectrum = reconstructed_spectrum[:, spectrum_mask]
        else:
            reconstructed_masked_spectrum = None

        outputs['reconstructed_spectrum'] = reconstructed_spectrum
        outputs['reconstructed_masked_spectrum'] = reconstructed_masked_spectrum
        if reduced_outputs is not None:
            outputs = reduce_model_outputs(outputs, reduced_outputs)

        return IRBertForMaskedSMOutput(**outputs)


class IRBertForMultiTaskPrediction(IRBertForPreTraining):
    def __init__(self, 
            config,
            tokenizer = None,
            processor = None,
            use_cache = True):

        if tokenizer is None:
            tokenizer = MultiTaskTokenizer(config = config)
        else:
            assert isinstance(tokenizer, MultiTaskTokenizer)

        super(IRBertForMultiTaskPrediction, self).__init__(
                config = config,
                tokenizer = tokenizer,
                processor = processor,
                use_cache = use_cache)

        task_weighting, task_tokens = {}, []
        for task_name in self.tokenizer.tasks:
            task_tokens += self.tokenizer.get_task_tokens(task_name)
            task_weighting[task_name] = nn.Paramter(torch.tensor(0.))

        self.task_weighting = nn.ParameterDict(task_weighting)

        tokens_head_size = {}
        for token in task_tokens:
            tokens_head_size[token] = self.tokenizer.get_token_head_size(token)

        self.task_predictor = MultiTaskPredictor(task_tokens = task_tokens,
                                                 tokens_head_size = tokens_head_size,
                                                 hidden_size = self.hidden_size,
                                                 dropout_prob = self.dropout_prob)

    def add_task(self, task_name, task_type, target_classes):
        self.tokenizer.add_task(task_name, task_type, target_classes)
        self.model.modify_token_embedding(self.tokenizer.token_number) 
        self.task_weighting[task_name] = nn.Parameter(torch.tensor(0.))
        
        for task_token in self.tokenizer.get_task_tokens(task_name):
            output_size = self.tokenizer.get_token_head_size(task_token)
            self.task_predictor.add_task_token(task_token, output_size)

        return None

    def delete_task(self, task_name):
        assert isinstance(task_name, str)
        assert task_name in self.tokenizer.tasks

        deleted_module_keys = []
        deleted_tokens = self.tokenizer.get_task_tokens(task_name)
        for token in deleted_tokens:
            if token in self.task_predictor.head_linear.keys():
                deleted_module_keys.append(token)

        replaced_tokens = self.tokenizer.content['tokens']
        token_indices = {}
        for token in replaced_tokens:
            token_indices[token] = self.tokenizer._find_token_id(token)

        self.tokenizer.delete_task(task_name, show_warning = False)

        replaced_token_indices = []
        tokens = self.tokenizer.content['tokens']
        for token in tokens:
            replaced_token_indices.append(token_indices[token])

        replaced_token_indices = torch.tensor(replaced_token_indices)
        self.model.modify_token_embedding(self.tokenizer.token_number,
                                          replaced_token_indices = replaced_token_indices)

        self.task_weighting.pop(task_name)
        for module_key in deleted_module_keys:
            self.task_predictor.delete_task_token(module_key)

        return None

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "IR-BERT's spectral embedding and fusion logic are not included in this public demo."
        )
        return IRBertForMultiTaskPredictionOutput(**outputs)

    def from_pretrained(cls,
            path,
            config_filename = 'config.json',
            tokenizer_filename = 'vocab.json',
            filename = 'model.safetensors'):

        assert isinstance(filename, str)
        assert len(filename) > 0
        if not filename.endswith('.safetensors'):
            filename += '.safetensors'

        self = super(IRBertForMultiTaskPrediction, cls).from_pretrained(path,
                                                                        config_filename = config_filename)

        self.tokenizer = self.tokenizer.from_pretrained(path,
                                                        filename = tokenizer_filename)

        for token in self.tokenizer.content['tokens_head_size']:
            self.task_predictor.add_task_token(token, self.tokenizer.content['tokens_head_size'][token])

        parameter_names = []
        for name, _ in self.named_parameters():
            parameter_names.append(name)

        model_filename = os.path.join(path, filename)
        state_dict = load_safetensors(model_filename)

        no_load_layers = [key for key in state_dict if key not in parameter_names]
        not_displayed_layers = []
        for layer_name in no_load_layers:
            if 'task_weighting.' in layer_name:
                not_displayed_layers.append(layer_name)

        if len(not_displayed_layers) > 0:
            for layer_name in not_displayed_layers:
                no_load_layers.remove(layer_name)

        if len(no_load_layers) > 0:
            message = 'Layers: {0} not contained in the {1}. '.format(no_load_layers, 
                    self.__class__.__name__) + 'If you are loading a pretrained weight,' + \
                    ' please ignore this warning message.'
            warnings.warn(message)

        filtered_state_dict = {k: v for k, v in state_dict.items() if k in parameter_names}
        self.load_state_dict(filtered_state_dict, strict = False)

        return self

    def save_pretrained(self,
            path,
            config_filename = 'config.json',
            tokenizer_filename = 'vocab.json',
            processor_filename = 'processor.safetensors',
            filename = 'model.safetensors'):

        handled_tasks = self.tokenizer.tasks
        self.config.update('handled_tasks', handled_tasks)

        super(IRBertForMultiTaskPrediction, self).save_pretrained(path = path,
                                                                  config_filename = config_filename,
                                                                  tokenizer_filename = tokenizer_filename,
                                                                  processor_filename = processor_filename,
                                                                  filename = filename)

        return None


