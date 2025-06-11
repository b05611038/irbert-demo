import os
import abc

import torch
import torch.nn as nn

from .config import IRBertConfig

__all__ = ['BaseModel']

class BaseModel(abc.ABC, nn.Module):
    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_pretrained(cls, path, config_filename = 'config.json'):
        assert isinstance(path, str)
        assert isinstance(config_filename, str)
        if not config_filename.endswith('.json'):
            config_filename += '.json'

        config = IRBertConfig()

        if os.path.isdir(path):
            config_filename = os.path.join(path, config_filename)
            if os.path.isfile(config_filename):
                config = config.from_json_file(config_filename)
            else:
                print('Cannot find valid model config in path, use default.')
        else:
            raise OSError('{0} is not a valid directory.'.format(path))

        return cls(config)

    @abc.abstractmethod
    def save_pretrained(self, path):
        raise NotImplementedError()


