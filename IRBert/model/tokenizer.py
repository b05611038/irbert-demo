import os
import abc
import warnings

from ..utils import (save_as_json, 
                     load_json, 
                     init_directory)


__all__ = ['IRBertTokenizer', 'MultiTaskTokenizer']


def task_content_handler(task_name, task_type, target_classes, 
        lower_case = True, existed_tasks = []):

    assert isinstance(task_name, str)
    assert len(task_name) > 0

    assert isinstance(task_type, str)
    assert task_type in ['multiclass_classification', 'multilabel_classification', 'regression']

    assert isinstance(target_classes, (tuple, list))
    for class_name in target_classes:
        if not isinstance(class_name, str):
            raise TypeError('Element in target_classes must be a Python string object.')

        if len(class_name) == 0:
            raise ValueError('Element in target_classes cannot be a empty string.')

    assert isinstance(lower_case, bool)
    assert isinstance(existed_tasks, (tuple, list))
    assert task_name not in existed_tasks
    if lower_case:
        task_name = task_name.lower()

    added_tokens, tokens_target, tokens_head_size = [], {}, {}
    if task_type == 'multiclass_classification':
        token = '[CLASSIFY] {0}'.format(task_name)
        added_tokens.append(token)
        tokens_target[token] = 'multiclass_classification'
        tokens_head_size[token] = len(target_classes)

    elif task_type == 'multilabel_classification':
        for class_name in target_classes:
            if lower_case:
                class_name = class_name.lower()

            token = '[CLASSIFY] {0} [IN] {1}'.format(class_name, task_name)
            added_tokens.append(token)
            tokens_target[token] = 'multilabel_classification'
            tokens_head_size[token] = 2

    elif task_type == 'regression':
         for class_name in target_classes:
            if lower_case:
                class_name = class_name.lower()

            token = '[REGRESS] {0} [IN] {1}'.format(class_name, task_name)
            added_tokens.append(token)
            tokens_target[token] = 'regression'
            tokens_head_size[token] = 1

    return added_tokens, tokens_target, tokens_head_size


class IRBertTokenizer:
    def __init__(self,
            vocab_file = None,
            config = None,
            mask_token = '[MASK]',
            pad_token = '[PAD]',
            sep_token = '[SEP]',
            use_cache = True):

        if config is not None:
            self.mask_token = config.get('mask_token', '[MASK]')
            self.pad_token = config.get('pad_token', '[PAD]')
            self.sep_token = config.get('sep_token', '[SEP]')
        else:
            self.mask_token = mask_token
            self.pad_token = pad_token
            self.sep_token = sep_token

        self.use_cache = use_cache
        self.tasks = []

        if vocab_file is None:
            self._content = {'tokens': [], 'tokens_target': {}}
            self.add_tokens([self.mask_token, self.pad_token, self.sep_token])
        else:
            self._content = self._load_vocab_file(vocab_file)

        self.__token_id_mapping = None
        if self.use_cache:
            self._construct_token_id_mapping()

    @property
    def content(self):
        return self._content

    @property
    def token_number(self):
        return len(self.content['tokens'])

    @property
    def use_cache(self):
        return self._use_cache

    @use_cache.setter
    def use_cache(self, use_cache):
        assert isinstance(use_cache, bool)
        self._use_cache = use_cache
        return None

    @property
    def mask_token_id(self):
        return self._find_token_id(self.mask_token)

    @property
    def sep_token_id(self):
        return self._find_token_id(self.sep_token)

    @property
    def pad_token_id(self):
        return self._find_token_id(self.pad_token)

    def _construct_token_id_mapping(self):
        self.__token_id_mapping = {}
        for idx in range(len(self.content['tokens'])):
            token = self.content['tokens'][idx]
            self.__token_id_mapping[token] = idx

        return None

    def _load_vocab_file(self, path, filename = 'vocab.json'):
        assert os.path.isdir(path) or os.path.isfile(path)
        if os.path.isdir(path):
            vocab_filename = os.path.join(path, filename)
        else:
            vocab_filename = path

        content = {}

        targeted_items = ['tokens', 'tokens_target']
        default_item = {'tokens': [],
                        'tokens_target': {}}

        loaded_content = load_json(vocab_filename)
        for item in targeted_items:
            content[item] = loaded_content.get(item, default_item[item])

        return content

    def __repr__(self):
        return '{0}(token_number={1})'.format(self.__class__.__name__, self.token_number)

    def _find_token_id(self, token):
        if self.use_cache:
            token_id = self.__token_id_mapping[token]
        else:
            token_id = self.content['tokens'].index(token)

        return token_id

    def __call__(self, tokenized_text):
        if isinstance(tokenized_text, str):
            raise ValueError('{0} not support text tokenizing.'.format(self.__class__.__name__))

        assert isinstance(tokenized_text, (tuple, list))
        if len(tokenized_text) > 0:
            token_ids = list(map(self._find_token_id, tokenized_text))
        else:
            token_ids = []

        return token_ids

    def add_tokens(self, tokens):
        assert isinstance(tokens, (str, tuple, list))
        if isinstance(tokens, str):
            tokens = [tokens]

        for t in tokens:
            assert t not in self.content['tokens']
            self.content['tokens'].append(t)
            self.content['tokens_target'][t] = 'model_retained'

        if self.use_cache:
            self._construct_token_id_mapping()

        return None

    def save_pretrained(self, path, filename = 'vocab.json'):
        assert isinstance(path, str)
        assert isinstance(filename, str)
        assert len(filename) > 0
        if not filename.endswith('.json'):
            filename += '.json'

        if os.path.isdir(path):
            config_filename = os.path.join(path, filename)
        else:
            if path.endswith(filename):
                config_filename = path
            else:
                init_directory(path)
                config_filename = os.path.join(path, filename)

        save_as_json(self._content, config_filename)

        return None

    def from_pretrained(self, path, filename = 'vocab.json'):
        assert isinstance(path, str)
        assert isinstance(filename, str)
        if not filename.endswith('.json'):
            filename += '.json'

        self._content = self._load_vocab_file(path, filename)
        if self.use_cache:
            self._construct_token_id_mapping()

        return self


class MultiTaskTokenizer(IRBertTokenizer):
    """
    Multi-task tokenizer for IR-BERT (public demo version).
    
    This module implements utility logic for assigning and managing tokens for
    multi-task predictions (e.g., regression, classification, masking tasks).
    
    Note:
    - This tokenizer does NOT process spectrum or wavelength data.
    - Core tokenization logic for spectral input is implemented in the processor module,
      which is excluded from this demo for confidentiality.
    """
    def __init__(self, 
            vocab_file = None, 
            config = None,
            mask_token = '[MASK]',
            pad_token = '[PAD]',
            sep_token = '[SEP]',
            tasks = (),
            tasks_contents = {},
            use_cache = True):

        if config is not None:
            self.mask_token = config.get('mask_token', '[MASK]')
            self.pad_token = config.get('pad_token', '[PAD]')
            self.sep_token = config.get('sep_token', '[SEP]')
        else:
            self.mask_token = mask_token
            self.pad_token = pad_token
            self.sep_token = sep_token

        assert isinstance(use_cache, bool)
        self.use_cache = use_cache

        assert isinstance(tasks_contents, dict)
        for task_name in tasks_contents:
            assert task_name in tasks

        if vocab_file is None:
            self._content = {'tasks': [], 
                             'tokens': [], 
                             'tokens_target': {}, 
                             'tokens_head_size': {},
                             'task_tokens': {}}
            self.add_tokens([self.mask_token, self.pad_token, self.sep_token])
        else:
            self._content = self._load_vocab_file(vocab_file)

        for task_name in tasks_contents:
            self.add_task(task_name, **tasks_contents[task_name])

        self.__token_id_mapping = None
        if self.use_cache:
            self._construct_token_id_mapping()

    def _load_vocab_file(self, path, filename = 'vocab.json'):
        assert os.path.isdir(path) or os.path.isfile(path)
        if os.path.isdir(path):
            vocab_filename = os.path.join(path, filename)
        else:
            vocab_filename = path

        content = {}

        targeted_items = ['tokens', 'tasks', 'tokens_target', 'tokens_head_size', 'task_tokens']
        default_item = {'tokens': [],
                        'tasks': [],
                        'tokens_target': {},
                        'tokens_head_size': {},
                        'task_tokens': {}}

        loaded_content = load_json(vocab_filename)
        for item in targeted_items:
            content[item] = loaded_content.get(item, default_item[item])

        return content

    @property
    def tasks(self):
        return self.content['tasks']

    def __repr__(self):
        return '{0}(token_number={1}, tasks={2})'.format(self.__class__.__name__,
                                                         self.token_number,
                                                         self.tasks)

    def get_tokenized_text(self, target_tasks, after_spectrum = True):
        assert isinstance(target_tasks, (str, tuple, list))
        if isinstance(target_tasks, str):
            target_tasks = [target_tasks]
        else:
            for task_name in target_tasks:
                assert task_name in self.tasks

        if after_spectrum:
            tokenized_text = [self.pad_token]
        else:
            tokenized_text = []

        for task_name in target_tasks:
            tokenized_text += self.get_task_tokens(task_name)
            tokenized_text.append(self.pad_token)

        return tokenized_text

    def add_task(self, task_name, task_type, target_classes):
        (tokens,
         tokens_target,
         tokens_head_size) = task_content_handler(task_name, 
                                                  task_type, 
                                                  target_classes)

        self.content['tasks'].append(task_name)
        self.content['tokens'] += tokens
        self.content['task_tokens'][task_name] = tokens
        for token in tokens_target:
            self.content['tokens_target'][token] = tokens_target[token]
            self.content['tokens_head_size'][token] = tokens_head_size[token]

        if self.use_cache:
            self._construct_token_id_mapping()

        return None

    def delete_task(self, task_name, show_warning = True):
        assert isinstance(task_name, str)
        assert task_name in self.tasks
        assert isinstance(show_warning, bool)

        if show_warning:
            warnings.warn("Method:delete_task is not recommended to be called in user's program." + \
                    " Dependencies with MTIRBertProcessor could be break. " + \
                    "Please use MTIRBertForTaskPrediction.delete_task for safe deletion of task.")

        task_tokens = self.get_task_tokens(task_name)
        for token in task_tokens:
            if token in self.content['tokens']:
                self.content['tokens'].remove(token)

            if token in self.content['tokens_target'].keys():
                del self.content['tokens_target'][token]

            if token in self.content['tokens_head_size'].keys():
                del self.content['tokens_head_size'][token]

        if task_name in self.content['task_tokens'].keys():
            del self.content['task_tokens'][task_name]

        if task_name in self.content['tasks']:
            self.content['tasks'].remove(task_name)

        if self.use_cache:
            self._construct_token_id_mapping()

        return None

    def get_token_target(self, token):
        assert isinstance(token, str)
        return self.content['tokens_target'][token]

    def get_token_head_size(self, token):
        assert isinstance(token, str)
        return self.content['tokens_head_size'][token]

    def get_task_type(self, task):
        assert isinstance(task, str)
        assert task in self.content['tasks']
        first_token = self.get_task_tokens(task)[0]
        task_type = self.get_token_target(first_token)
        return task_type

    def get_task_tokens(self, task):
        assert isinstance(task, str)
        assert task in self.content['task_tokens'].keys()
        return self.content['task_tokens'][task]


