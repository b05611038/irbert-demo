import abc
import copy
import json

from .file import save_as_json, load_json


__all__ = ['Configuration']


class Configuration(abc.ABC):
    def __init__(self, default_config = None, **kwargs):
        if default_config is None:
            default_config = {'complementary': {}}
            for args_key in kwargs:
                default_config[args_key] = kwargs[args_key]

        assert isinstance(default_config, dict)
        self.__default_config = default_config
        self.__param_dict = {}

        for args_key in kwargs:
            assert args_key != 'complementary'
            if args_key not in default_config.keys():
                raise KeyError('Argument: {0} is not valid for {1}.'\
                        .format(args_key, self.__class__.__name__))

            self.__param_dict[args_key] = None

        for args_key in kwargs:
            self.update(args_key, kwargs[args_key])

    @property
    def config(self):
        return self.to_dict()

    @property
    def default(self):
        config = copy.deepcopy(self.__default_config)
        return config

    def __repr__(self):
        text = '{0}('.format(self.__class__.__name__)
        reduction = False
        for args_key in self.__param_dict:
            if isinstance(self.__param_dict[args_key], str):
                text += '"{0}"="{1}" ,'.format(args_key, self.__param_dict[args_key])
            else:
                text += '"{0}"={1}, '.format(args_key, self.__param_dict[args_key])

            reduction = True

        if reduction:
            text = text[: -2]

        text += ')'
        return text

    def from_dict(self, config_dict):
        assert isinstance(config_dict, dict)
        for key in config_dict:
            if key == 'complementary':
                for module_name in config_dict['complementary']:
                    self.add_extra_config(module_name, config_dict['complementary'][module_name])
            else:
                self.update(key, config_dict[key])

        return self

    def from_json_file(self, json_file, extension_check = True):
        config_dict = load_json(json_file, extension_check = extension_check)
        for key in config_dict:
            if key == 'complementary':
                for module_name in config_dict['complementary']:
                    self.add_extra_config(module_name, config_dict['complementary'][module_name])
            else:
                self.update(key, config_dict[key])

        return self

    def to_dict(self):
        config = {}
        for args_key in self.__param_dict:
            config[args_key] = copy.deepcopy(self.__param_dict[args_key])

        return config

    def to_diff_dict(self):
        diff_config = {}
        for args_key in self.__param_dict:
            if args_key == 'complementary':
                continue

            if self.__param_dict[args_key] != self.__default_config[args_key]:
                diff_config[args_key] = self.__param_dict[args_key]

        return diff_config

    def to_json_file(self, filename):
        config = self.to_dict()
        save_as_json(config, filename)
        return None

    def to_json_string(self, indent = None):
        json_string = None
        config = self.to_dict()
        if indent is not None:
            assert isinstance(indent, int)
            assert indent >= 0
            json_string = json.dumps(config, indent = indent)
        else:
            json_string = json.dumps(config)

        return json_string

    def get(self, args_key, default_value = None):
        assert isinstance(args_key, str)
        if args_key in self.__param_dict.keys():
            value = self.__param_dict[args_key]
        else:
            value = default_value

        return value

    def update(self, args_key, argument):
        assert isinstance(args_key, str)
        if argument is not None:
            assert isinstance(argument, (int, float, str, tuple, list))

        if args_key not in self.__param_dict.keys():
            raise KeyError('Argument: {0} is not valid for {1}.'\
                    .format(args_key, self.__class__.__name__))

        self.__param_dict[args_key] = argument

        return None

    def summary(self, pad_space = 2, system_out = print):
        assert isinstance(pad_space, int)
        assert pad_space >= 0

        text = '{0}(\n'.format(self.__class__.__name__)
        for args_key in self.__param_dict:
            text += ' ' * pad_space
            if isinstance(self.__param_dict[args_key], str):
                text += '"{0}": "{1}",\n'.format(args_key, self.__param_dict[args_key])
            else:
                text += '"{0}": {1},\n'.format(args_key, self.__param_dict[args_key])

        text += ')'
        system_out(text)
        return None

    def add_extra_config(self, module_name, config):
        assert isinstance(module_name, str)
        assert isinstance(config, dict)

        if 'complementary' not in self.__param_dict:
            self.__param_dict['complementary'] = {}

        self.__param_dict[module_name] = config
        return None

    def get_extra_config(self, module_name):
        assert isinstance(module_name, str)
        assert module_name in self.__param_dict['complementary'].keys()
        config = self.__param_dict['complementary'][module_name]
        return config


