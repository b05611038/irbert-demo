import os
import random
import string
import copy

import numpy as np

import torch

coffee_database_available = True
try:
    import coffee_database
except ImportError:
    coffee_database_available = False

if coffee_database_available:
    from coffee_database import (CoffeeDatabase,
                                 CoffeeDatabaseDataset,
                                 EnsembleCoffeeDatabaseDataset)

    from coffee_database.dataset import _split_coffee_samples

from .config import (dataset_configuration,
                     ImplementedTasks)

from .resample import resample_spectrum
from ..utils import save_as_json

__all__ = ['select_single_dataset', 'construct_ensemble_dataset', 
        'get_default_dataset_content', 'MultiTaskingEnsembleDataset']

CoffeeSampleSelection = '[NIR] {field: with}'
DefaultTaskContent = {
        'agtron_regression': {'task_type': 'regression',
                              'target_classes': ['agtron_value']},

        'cupping_report_regression': {'task_type': 'regression',
                                      'target_classes': ['fragrance/aroma',
                                                         'flavor',
                                                         'aftertastes',
                                                         'acidity',
                                                         'body',
                                                         'balance',
                                                         'overall']},

        'flavor_classification': {'task_type': 'multilabel_classification',
                                  'target_classes': ['floral',
                                                     'fruity',
                                                     'sour/fermented',
                                                     'green/vegetative',
                                                     'other',
                                                     'roasted',
                                                     'spices',
                                                     'nutty/cocoa',
                                                     'sweet']},
        'flavor_regression': {'task_type': 'regression',
                              'target_classes': ['floral',
                                                 'fruity',
                                                 'sour/fermented',
                                                 'green/vegetative',
                                                 'other',
                                                 'roasted',
                                                 'spices',
                                                 'nutty/cocoa',
                                                 'sweet']},

        'country_classification': {'task_type': 'multiclass_classification',
                                   'target_classes': ['Ethiopia',
                                                      'Columbia',
                                                      'Costa Rica',
                                                      'Guatemala',
                                                      'Brazil',
                                                      'El Salvador',
                                                      'Honduras',
                                                      'Indonesia',
                                                      'Kenya',
                                                      'Panama',
                                                      'Nicaragua',
                                                      'Peru',
                                                      'Yemen',
                                                      'Malawi',
                                                      'unknown', # blend samples
                                                      'other']},

        'process_method_classification': {'task_type': 'multiclass_classification',
                                          'target_classes': ['washed',
                                                             'natural',
                                                             'honey',
                                                             'other']},
}


class MultiTaskingEnsembleDataset(EnsembleCoffeeDatabaseDataset):
    def __init__(self, 
            datasets, 
            split_config = {'mode': 'train',
                            'split_ratio': {'train': 0.64, 'val': 0.16, 'test': 0.2},
                            'seed': None,
                            'val_seed': None},
            transform = None,
            test_transform = None,
            label_transform = None,
            test_label_transform = None,
            mini_batch_size = None,
            matrix_datatype = 'torch'):

        self.use_inheritted_api = True

        super(MultiTaskingEnsembleDataset, self).__init__(
                datasets = datasets,
                ensemble_method = 'intersection',
                random_sampling = False,
                split_config = split_config,
                transform = transform,
                test_transform = test_transform,
                label_transform = label_transform,
                test_label_transform = test_label_transform,
                mini_batch_size = mini_batch_size,
                matrix_datatype = matrix_datatype)

        if len(self.tasks) > 1:
            first_task = self.tasks[0]
            cheched_tasks = self.tasks[1: ]
            for task in cheched_tasks:
                assert self.transform[first_task] == self.transform[task]
                assert self.test_transform[first_task] == self.test_transform[task]

            for sample in self.data[first_task].keys():
                for task in cheched_tasks:
                    if not np.allclose(self.data[first_task][sample],
                                       self.data[task][sample],
                                       atol = 1e-8):

                        raise RuntimeError('Detect difference data inputted in different tasks.')

        self.use_inheritted_api = False

    @property
    def task_now(self):
        if self.use_inheritted_api:
            return super(MultiTaskingEnsembleDataset, self).task_now
        else:
            return self.tasks

    def __len__(self):
        return self.data_number[self.tasks[0]][self.mode]

    def __repr__(self):
        lines = '{0}(tasks={1})\n'.format(self.__class__.__name__, self.tasks)
        lines += '\n  Operation mode: {0}\n  Data number in this mode: {1}\n  Data shape: {2}\n'\
                .format(self.mode, len(self), self.displayed_data_shape[self.tasks[0]])

        for task in self.tasks:
            lines += '  Label shape ({0}): {1}\n'.format(task, self.displayed_label_shape[task])

        if self.mode == 'val' or self.mode == 'test':
            if self.test_transform[self.tasks[0]] is not None:
                lines += '  Data Transform:\n  {0}'.format(self.test_transform[self.task_now])
        else:
            if self.transform[self.tasks[0]] is not None:
                lines += '  Data Transform:\n  {0}'.format(self.transform[self.task_now])

        for task in self.tasks:
            if self.test_label_transform[task] is not None:
                lines += '  Label Transform (task={0}):\n  {1}'.format(task,
                        self.test_label_transform[task])

        return lines

    def __getitem__(self, index):
        if self.use_inheritted_api:
            return super(MultiTaskingEnsembleDataset, self).__getitem__(index = index)
        else:
            data = self.data_matrix[self.tasks[0]][self.mode][index]
            if self.mode == 'val' or self.mode == 'test':
                if self.test_transform[self.tasks[0]] is not None:
                    data = self.test_transform[self.tasks[0]](data)
            else:
                if self.transform[self.tasks[0]] is not None:
                    data = self.transform[self.tasks[0]](data) 

            labels = []
            for task in self.tasks:
                label = self.label_matrix[task][self.mode][index]
                if self.mode == 'val' or self.mode == 'test':
                    if self.test_label_transform[task] is not None:
                        label = self.self.test_label_transform[task](label)
                else:
                    if self.label_transform[task] is not None:
                        label  = self.label_transform[task](label)

                labels.append(label)

            return (data, *labels)

    def set_task(self, task):
        if self.use_inheritted_api:
            super(MultiTaskingEnsembleDataset, self).set_task(task)
        else:
            raise RuntimeError('Method:set_task is not support for {0}'\
                    .format(self.__class__.__name__))

        return None

    def random_switch_task(self):
        raise RuntimeError('Method:random_switch_task is not support for {0}'\
                .format(self.__class__.__name__))

    def data_size(self):
        return self.displayed_data_shape[self.tasks[0]]

    def label_size(self):
        return copy.deepcopy(self.displayed_label_shape)

    def catch_all(self):
        self.use_inheritted_api = True
        outputs, first = [], True
        for task in self.tasks:
            data, label = super(MultiTaskingEnsembleDataset, self).catch_all(task = task)
            if first:
                outputs.append(data)
                first = False

            outputs.append(label)

        self.use_inheritted_api = False

        return tuple(outputs)

    def set_transform(self, transform, task = None, label = False):
        assert isinstance(label, bool)
        if label:
             super(MultiTaskingEnsembleDataset, self).set_transform(transform = transform,
                     task = task, label = label)
        else:
             super(MultiTaskingEnsembleDataset, self).set_transform(transform = transform,
                     task = self.tasks[0], label = label)

        return None

    def set_test_transform(self, transform, task = None, label = False):
        assert isinstance(label, bool)
        if label:
            super(MultiTaskingEnsembleDataset, self).set_test_transform(transform = transform,
                    task = task, label = label)
        else:
            super(MultiTaskingEnsembleDataset, self).set_test_transform(transform = transform,
                     task = self.tasks[0], label = label)

        return None

    def acquire_origin_sample_data(self):
        outputs, fisrt = [], True
        for task in self.tasks:
             (data, 
              label) = super(MultiTaskingEnsembleDataset, self).acquire_origin_sample_data(task)

             if fisrt:
                 outputs.append(data)
                 first = False

             outputs.append(label)

        return tuple(outputs)

    def label_distribution(self, mode = None):
        collect_label, collect_header = {}, {}
        for task in self.task:
            (label, 
             header) = super(MultiTaskingEnsembleDataset, self).label_distribution(task = task, 
                                                                                   mode = mode)

            collect_label[task] = label
            collect_header[task] = header

        return collect_label, collect_header

    def summary_label_distribution(self, mode = None, ratio = True,
            category_length = 15, screen_length = 120, label_display_ranges = None):

        for task in self.tasks:
            super(MultiTaskingEnsembleDataset, self).summary_label_distribution(task = task,
                    mode = mode, ratio = ratio, category_length = category_length,
                    screen_length = screen_length, label_display_ranges = label_display_ranges)

        return None


def select_single_dataset(task, country_number = None, seed = 0, val_seed = 0, 
        samples = CoffeeSampleSelection, split_ratio = None, mean_spectrum_mode = False, 
        sample_level_split = True, resampling = False, drop_spectrum_indices = None):

    if not coffee_database_available:
        raise ImportError('The extended private library: coffee_database does not installed.')

    assert isinstance(task, str)
    assert task in ImplementedTasks
    if country_number is not None:
        assert isinstance(country_number, int)
        assert country_number > 0 and country_number <= 14

    if split_ratio is not None:
        assert isinstance(split_ratio, dict)

    assert isinstance(mean_spectrum_mode, bool)
    assert isinstance(sample_level_split, bool)
    assert isinstance(resampling, bool)
    if drop_spectrum_indices is not None:
        assert isinstance(drop_spectrum_indices, (tuple, list))
        for drop_idx in drop_spectrum_indices:
            assert isinstance(drop_idx, int)

    db = CoffeeDatabase()
    all_coffee_samples_in_db = db.filter_sample_by_config('')

    dataset_config = dataset_configuration(task = task,
                                           seed = seed,
                                           val_seed = val_seed,
                                           split_ratio = split_ratio,
                                           mean_spectrum_mode = mean_spectrum_mode,
                                           samples = samples)

    if country_number is not None:
        if 'country_classification' in task:
            blend_text = dataset_config['label_config']['argument']['groups'][-1]
            replaced_groups = dataset_config['label_config']['argument']['groups'][: country_number]
            replaced_groups.append(blend_text)
            dataset_config['label_config']['argument']['groups'] = replaced_groups

    datatype = dataset_config['data_config']['datatype']
    wavelengths = db.axis_label_of_detecting_data(datatype)

    data_name = list(wavelengths.keys())[0]
    wavelengths = wavelengths[data_name]
    wavelengths = torch.tensor(wavelengths, dtype = torch.float)

    dataset = CoffeeDatabaseDataset(**dataset_config)
    dataset.wavelengths = wavelengths

    if 'reconstruction' in task:
        all_samples = dataset.samples['train'] + dataset.samples['val'] + dataset.samples['test'] 
        coffee_samples = {}
        for sample_name in all_samples:
            original_name = copy.copy(sample_name)
            for i in range(1, 6):
                search_pattern = '_{0}'.format(i)
                if sample_name.endswith(search_pattern):
                    original_name = sample_name[: -2]
                    break

            if original_name not in coffee_samples.keys():
                coffee_samples[original_name] = [sample_name]
            else:
                coffee_samples[original_name].append(sample_name)

        split_ratio = dataset.split_ratio
        seed = dataset.seed
        val_seed = dataset.val_seed

        meta_dataset_split = _split_coffee_samples(list(coffee_samples.keys()),
                split_ratio, seed, val_seed)

        dataset_split = {'train': [], 'val': [], 'test': []}
        for set_name in meta_dataset_split:
            for meta_sample_name in meta_dataset_split[set_name]:
                dataset_split[set_name] += coffee_samples[meta_sample_name]

        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k = 15))
        rand_filename = '.' + random_string + '.json'
        save_as_json(dataset_split, rand_filename)
        dataset.load_split(fname = rand_filename, skip_shortage = False)
        os.remove(rand_filename)

    if resampling:
        target_wavelengths = db.axis_label_of_detecting_data('ITRI_NIR')['ITRI_nir_spectrum'][1: ]
        target_wavelengths = torch.tensor(target_wavelengths, dtype = torch.float)
        for sample_name in dataset.data:
            sample_spectra = torch.tensor(dataset.data[sample_name], dtype = torch.float)
            dataset.data[sample_name] = resample_spectrum(sample_spectra,
                                                          wavelengths,
                                                          target_wavelengths)


        dataset.padded_data_shape = target_wavelengths.shape
        dataset.displayed_data_shape = target_wavelengths.shape
        dataset.split_samples()
        dataset.wavelengths = target_wavelengths

    if drop_spectrum_indices is not None:
        target_wavelengths = dataset.wavelengths
        retained_indices = [i for i in range(target_wavelengths.shape[0])]
        for idx in drop_spectrum_indices:
            if idx in retained_indices:
                retained_indices.remove(idx)

        retained_indices = torch.tensor(retained_indices)
        target_wavelengths = target_wavelengths[retained_indices]
        for sample_name in dataset.data:
            if not isinstance(dataset.data[sample_name], torch.Tensor):
                dataset.data[sample_name] =  torch.tensor(dataset.data[sample_name], 
                        dtype = torch.float)

            dataset.data[sample_name] = dataset.data[sample_name][:, retained_indices]

        dataset.padded_data_shape = target_wavelengths.shape
        dataset.displayed_data_shape = target_wavelengths.shape
        dataset.split_samples()
        dataset.wavelengths = target_wavelengths

    return dataset

def construct_ensemble_dataset(tasks, country_number = None, seed = 0, val_seed = 0, 
        samples = CoffeeSampleSelection, switch_mode = 'random_sample', 
        sample_level_split = True, resampling = False, drop_spectrum_indices = None):

    if not coffee_database_available:
        raise ImportError('The extended private library: coffee_database does not installed.')

    assert isinstance(tasks, list)
    if country_number is not None:
        assert isinstance(country_number, int)
        assert country_number > 0 and country_number <= 14

    assert isinstance(seed, int)
    assert isinstance(val_seed, int)
    assert isinstance(samples, str)
    assert isinstance(switch_mode, str)
    available_switch = ['random_sample', 'together']
    if switch_mode not in available_switch:
        raise ValueError('{0} is invalid for switch_mode (only accept {1}).'\
                .format(switch_mode, available_switch))

    assert isinstance(sample_level_split, bool)
    assert isinstance(resampling, bool)
    if drop_spectrum_indices is not None:
        assert isinstance(drop_spectrum_indices, (tuple, list))
        for drop_idx in drop_spectrum_indices:
            assert isinstance(drop_idx, int)

    db = CoffeeDatabase()
    for task in tasks:
        assert task in ImplementedTasks

    datasets, tasks_wavelengths = {}, {}
    for task in tasks:
        print('Initialize task: {0} ...'.format(task))
        task_config = dataset_configuration(task = task,
                                            seed = seed,
                                            val_seed = val_seed,
                                            samples = samples)

        if country_number is not None:
            if 'country_classification' in task:
                blend_text = task_config['label_config']['argument']['groups'][-1]
                replaced_groups = task_config['label_config']['argument']['groups'][: country_number]
                replaced_groups.append(blend_text)
                task_config['label_config']['argument']['groups'] = replaced_groups

        datatype = task_config['data_config']['datatype']
        wavelengths = db.axis_label_of_detecting_data(datatype)

        data_name = list(wavelengths.keys())[0]
        wavelengths = wavelengths[data_name]
        wavelengths = torch.tensor(wavelengths, dtype = torch.float)

        single_dataset = CoffeeDatabaseDataset(**task_config)
        single_dataset.wavelengths = wavelengths

        tasks_wavelengths[task] = single_dataset.wavelengths
        if 'reconstruction' in task:
            all_samples = single_dataset.samples['train'] + single_dataset.samples['val'] + \
                    single_dataset.samples['test']

            coffee_samples = {}
            for sample_name in all_samples:
                original_name = copy.copy(sample_name)
                for i in range(1, 6):
                    search_pattern = '_{0}'.format(i)
                    if sample_name.endswith(search_pattern):
                        original_name = sample_name[: -2]
                        break

                if original_name not in coffee_samples.keys():
                    coffee_samples[original_name] = [sample_name]
                else:
                    coffee_samples[original_name].append(sample_name)

            split_ratio = single_dataset.split_ratio
            seed = single_dataset.seed
            val_seed = single_dataset.val_seed

            meta_dataset_split = _split_coffee_samples(list(coffee_samples.keys()),
                    split_ratio, seed, val_seed)

            dataset_split = {'train': [], 'val': [], 'test': []}
            for set_name in meta_dataset_split:
                for meta_sample_name in meta_dataset_split[set_name]:
                    dataset_split[set_name] += coffee_samples[meta_sample_name]

            characters = string.ascii_letters + string.digits
            random_string = ''.join(random.choices(characters, k = 15))
            rand_filename = '.' + random_string + '.json'
            save_as_json(dataset_split, rand_filename)
            single_dataset.load_split(fname = rand_filename, skip_shortage = False)
            os.remove(rand_filename)

        if resampling:
            target_wavelengths = db.axis_label_of_detecting_data('ITRI_NIR')['ITRI_nir_spectrum'][1: ]
            target_wavelengths = torch.tensor(target_wavelengths, dtype = torch.float)
            tasks_wavelengths[task] = target_wavelengths

            for sample_name in single_dataset.data:
                sample_spectra = torch.tensor(single_dataset.data[sample_name], dtype = torch.float)
                single_dataset.data[sample_name] = resample_spectrum(sample_spectra,
                                                                     wavelengths,
                                                                     target_wavelengths)

                single_dataset.padded_data_shape = target_wavelengths.shape
                single_dataset.displayed_data_shape = target_wavelengths.shape
                single_dataset.split_samples()
                single_dataset.wavelengths = target_wavelengths

        if drop_spectrum_indices is not None:
            target_wavelengths = dataset.wavelengths
            retained_indices = [i for i in range(target_wavelengths.shape[0])]
            for idx in drop_spectrum_indices:
                if idx in retained_indices:
                    retained_indices.remove(idx)

            retained_indices = torch.tensor(retained_indices)
            target_wavelengths = target_wavelengths[retained_indices]
            for sample_name in single_dataset.data:
                if not isinstance(single_dataset.data[sample_name], torch.Tensor):
                    single_dataset.data[sample_name] =  torch.tensor(single_dataset.data[sample_name],
                            dtype = torch.float)

                single_dataset.data[sample_name] = single_dataset.data[sample_name][:, retained_indices]

            single_dataset.padded_data_shape = target_wavelengths.shape
            single_dataset.displayed_data_shape = target_wavelengths.shape
            single_dataset.split_samples()
            single_dataset.wavelengths = target_wavelengths

        datasets[task] = single_dataset

    ensemble_dataset = None
    if switch_mode == 'random_sample':
        ensemble_dataset = EnsembleCoffeeDatabaseDataset(datasets)
        ensemble_dataset.wavelengths = tasks_wavelengths
    elif switch_mode == 'together':
        ensemble_dataset = MultiTaskingEnsembleDataset(datasets)
        ensemble_dataset.wavelengths = tasks_wavelengths[list(tasks_wavelengths.keys())[0]]

    print('Construct the ensemble dataset successfully.')

    return ensemble_dataset

def get_default_dataset_content(task):
    if not coffee_database_available:
        raise ImportError('The extended private library: coffee_database does not installed.')

    assert isinstance(task, str)
    assert task in ImplementedTasks

    match_task = None
    for task_label in DefaultTaskContent:
        if task_label in task:
            match_task = task_label

    if match_task is None:
        raise ValueError('Task: {0} not implemented now.'.format(task))

    return copy.deepcopy(DefaultTaskContent[match_task])


