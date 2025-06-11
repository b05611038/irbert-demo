import copy

__all__ = ['dataset_configuration', 'ImplementedTasks']

DefaultSeed = 0
DefaultValSeed = 0
DefaultFlavorCategory = 'inner'

DefaultDatabaseConfig = {
        'db_name': 'coffee',
        'user_id': 'coffee_admin',
        'passwd': None,
        'host': 'localhost',
        'port': 27017,
}

DefaultSampleConfig = {'program': '[NIR] {field: with}'}

DefaultFinetuneSplitConfig = {'seed': copy.deepcopy(DefaultSeed),
                              'val_seed': copy.deepcopy(DefaultValSeed),
                              'split_ratio': {'train': 0.64, 'val': 0.16, 'test': 0.2}}

DefaultPretrainSplitConfig = {'seed': copy.deepcopy(DefaultSeed),
                              'val_seed': copy.deepcopy(DefaultValSeed),
                              'split_ratio': {'train': 0.8, 'val': 0., 'test': 0.2}}

DefaultNIRDataConfig = {'datatype': 'NIR',
                        'index_alignment': 'index_select',
                        'target_axis': None,
                        'mean': False,
                        'reduce_blank': False}

DefaultITRINIRDataConfig = {'datatype': 'ITRI_NIR',
                            'index_alignment': 'index_select',
                            'target_axis': None,
                            'mean': False,
                            'reduce_blank': False}

DefaultFTIRDataConfig = {'datatype': 'FTIR',
                         'index_alignment': 'index_select',
                         'target_axis': None,
                         'mean': False,
                         'reduce_blank': False}

DefaultCuppingReportDataConfig = {
        'datatype': 'cupping_report_and_flavor',
        'scoring_items': {'drop_nan_any_sample': False,
                          'drop_total_score': True,
                          'drop_cross_cup_items': True,
                          'drop_intensity_items': True,
                          'merge_mode': 'average',
                          'rescaling': {'columns': ['dry', 'break', 'acidity_level', 'body_level'],
                                        'mode': 'linear_scale',
                                        'scaled_minimum': 0.,
                                        'scaled_maximum': 1.,
                                        'scaled_mean': 0.,
                                        'scaled_std': 1.}},
        'descriptors': {'description': {'strength_threshold': 0.01,
                                        'strength_padding': 3.,
                                        'merge_mode': 'largest'},
                        'flavor_wheel': {'layer': copy.deepcopy(DefaultFlavorCategory),
                                         'binary': False},
                        'numericalize_after_merge': True},
}

DefaultAgtronRegressionLabelConfig = {
        'labeltype': 'agtron',
        'argument': {'agtron_type': 'ground'}
}

DefaultBlendingCuppingReportRegressionLabelConfig = {
        'labeltype': 'blend_coffee',
        'argument': copy.deepcopy(DefaultCuppingReportDataConfig),
}

DefaultCuppingReportRegressionLabelConfig = {
        'labeltype': 'cupping_report',
        'argument': copy.deepcopy(DefaultCuppingReportDataConfig),
}

DefaultDataReconstructionLabelConfig = {
        'labeltype': 'data_reconstruction',
        'argument': None,
}

DefaultFlavorClassificationLabelConfig = {
        'labeltype': 'description',
        'argument': {'description': {'description_type': 'flavor',
                                     'strength_threshold': 0.01,
                                     'strength_padding': 3.,
                                     'merge_mode': 'largest'},
                     'flavor_wheel': {'layer': copy.deepcopy(DefaultFlavorCategory),
                                      'binary': True},
                     'numericalize_after_merge': True,
                     'remove_empty_sample': False},
}

DefaultFlavorRegressionLabelConfig = {
        'labeltype': 'description',
        'argument': {'description': {'description_type': 'flavor',
                                     'strength_threshold': 0.01,
                                     'strength_padding': 3.,
                                     'merge_mode': 'largest'},
                     'flavor_wheel': {'layer': copy.deepcopy(DefaultFlavorCategory),
                                      'binary': False},
                     'numericalize_after_merge': True,
                     'remove_empty_sample': False},
}

DefaultCountryClassificationLabelConfig = {
        'labeltype': 'group_by_expressions',
        'argument': {'start_index': 0,
                     'label_datatype': 'int',
                     'others_usage': True,
                     'groups': ['[ELEMENT] {country: Ethiopia}',
                                '[ELEMENT] {country: Colombia}',
                                '[ELEMENT] {country: Costa Rica}',
                                '[ELEMENT] {country: Guatemala}',
                                '[ELEMENT] {country: Brazil}',
                                '[ELEMENT] {country: El Salvador}',
                                '[ELEMENT] {country: Honduras}',
                                '[ELEMENT] {country: Indonesia}',
                                '[ELEMENT] {country: Kenya}',
                                '[ELEMENT] {country: Panama}',
                                '[ELEMENT] {country: Nicaragua}',
                                '[ELEMENT] {country: Peru}',
                                '[ELEMENT] {country: Yemen}',
                                '[ELEMENT] {country: Malawi}',
                                '[ELEMENT] {country: unknown}']},
}

DefaultProcessMethodClassificationLabelConfig = {
        'labeltype': 'group_by_expressions',
        'argument': {'start_index': 0, 
                     'label_datatype': 'int',
                     'others_usage': True,
                     'groups': ['[ELEMENT] {process: washed}',
                                '[ELEMENT] {process: natural}',
                                '[ELEMENT] {process: honey}']},
}

def dataset_configuration(task, seed = DefaultSeed, val_seed = DefaultValSeed, 
        split_ratio = None, mean_spectrum_mode = False, samples = '[NIR] {field: with}'):

    assert isinstance(task, str)
    assert isinstance(seed, int)
    assert isinstance(val_seed, int)
    if split_ratio is not None:
        assert isinstance(split_ratio, dict)
        necessary_keys = ['train', 'val', 'test']
        for key in necessary_keys:
            if key not in split_ratio.keys():
                raise KeyError('Key: {0} must exist in dict::split_ratio.')

            assert isinstance(split_ratio[key], float)
            assert split_ratio[key] >= 0. and split_ratio[key] <= 1.

    assert isinstance(mean_spectrum_mode, bool)
    assert isinstance(samples, str)

    if 'mask_reconstruction' in task or 'data_reconstruction' in task:
        split_config = copy.deepcopy(DefaultPretrainSplitConfig)
    else:
        split_config = copy.deepcopy(DefaultFinetuneSplitConfig)

    split_config['seed'] = seed
    split_config['val_seed'] = val_seed
    if split_ratio is not None:
        split_config['split_ratio'] = split_ratio

    dataset_config = {'database_config': copy.deepcopy(DefaultDatabaseConfig),
                      'sample_config': {'program': samples},
                      'split_config': split_config}

    input_task = copy.deepcopy(task)
    task = task.lower()

    if task == 'nir_data_reconstruction' or task == 'nir_mask_reconstruction':
        dataset_config['data_config'] = copy.deepcopy(DefaultNIRDataConfig)
        label_config = copy.deepcopy(DefaultDataReconstructionLabelConfig)
        label_config['argument'] = copy.deepcopy(DefaultNIRDataConfig)
        dataset_config['label_config'] = label_config
    elif task == 'itri_nir_data_reconstruction' or task == 'itri_nir_mask_reconstruction':
        dataset_config['data_config'] = copy.deepcopy(DefaultITRINIRDataConfig)
        label_config = copy.deepcopy(DefaultDataReconstructionLabelConfig)
        label_config['argument'] = copy.deepcopy(DefaultITRINIRDataConfig)
        dataset_config['label_config'] = label_config
    elif task == 'ftir_data_reconstruction' or task == 'ftir_mask_reconstruction':
        dataset_config['data_config'] = copy.deepcopy(DefaultFTIRDataConfig)
        label_config = copy.deepcopy(DefaultDataReconstructionLabelConfig)
        label_config['argument'] = copy.deepcopy(DefaultFTIRDataConfig)
        dataset_config['label_config'] = label_config
    elif task == 'nir_agtron_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultNIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultAgtronRegressionLabelConfig)
    elif task == 'itri_nir_agtron_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultITRINIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultAgtronRegressionLabelConfig)
    elif task == 'ftir_agtron_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultFTIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultAgtronRegressionLabelConfig)
    elif task == 'nir_blend_cupping_report_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultNIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultBlendingCuppingReportRegressionLabelConfig)
    elif task == 'itri_nir_blend_cupping_report_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultITRINIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultBlendingCuppingReportRegressionLabelConfig)
    elif task == 'ftir_blend_cupping_report_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultFTIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultBlendingCuppingReportRegressionLabelConfig)
    elif task == 'nir_cupping_report_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultNIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultCuppingReportRegressionLabelConfig)
    elif task == 'itri_nir_cupping_report_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultITRINIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultCuppingReportRegressionLabelConfig)
    elif task == 'ftir_cupping_report_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultFTIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultCuppingReportRegressionLabelConfig)
    elif task == 'nir_flavor_classification':
        dataset_config['data_config'] = copy.deepcopy(DefaultNIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultFlavorClassificationLabelConfig)
    elif task == 'itri_nir_flavor_classification':
        dataset_config['data_config'] = copy.deepcopy(DefaultITRINIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultFlavorClassificationLabelConfig)
    elif task == 'ftir_flavor_classification':
        dataset_config['data_config'] = copy.deepcopy(DefaultFTIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultFlavorClassificationLabelConfig)
    elif task == 'nir_flavor_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultNIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultFlavorRegressionLabelConfig)
    elif task == 'itri_nir_flavor_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultITRINIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultFlavorRegressionLabelConfig)
    elif task == 'ftir_flavor_regression':
        dataset_config['data_config'] = copy.deepcopy(DefaultFTIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultFlavorRegressionLabelConfig)
    elif task == 'nir_country_classification':
        dataset_config['data_config'] = copy.deepcopy(DefaultNIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultCountryClassificationLabelConfig)
    elif task == 'itri_nir_country_classification':
        dataset_config['data_config'] = copy.deepcopy(DefaultITRINIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultCountryClassificationLabelConfig)
    elif task == 'ftir_country_classification':
        dataset_config['data_config'] = copy.deepcopy(DefaultFTIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultCountryClassificationLabelConfig)
    elif task == 'nir_process_method_classification':
        dataset_config['data_config'] = copy.deepcopy(DefaultNIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultProcessMethodClassificationLabelConfig)
    elif task == 'itri_nir_process_method_classification':
        dataset_config['data_config'] = copy.deepcopy(DefaultITRINIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultProcessMethodClassificationLabelConfig)
    elif task == 'ftir_process_method_classification':
        dataset_config['data_config'] = copy.deepcopy(DefaultFTIRDataConfig)
        dataset_config['label_config'] = copy.deepcopy(DefaultProcessMethodClassificationLabelConfig)
    else:
        raise NotImplementedError('Task: {0} is not implemented in the source code'.format(input_task))

    if mean_spectrum_mode:
        dataset_config['data_config']['mean'] = True

    return dataset_config

ImplementedTasks = ['nir_data_reconstruction', 'itri_nir_data_reconstruction', 'ftir_data_reconstruction', 
                    'nir_mask_reconstruction', 'itri_nir_mask_reconstruction', 'ftir_mask_reconstruction', 
                    'nir_agtron_regression', 'itri_nir_agtron_regression', 'ftir_agtron_regression', 
                    'nir_blend_cupping_report_regression', 'itri_nir_blend_cupping_report_regression', 
                    'ftir_blend_cupping_report_regression', 'nir_cupping_report_regression', 
                    'itri_nir_cupping_report_regression', 'ftir_cupping_report_regression', 
                    'nir_flavor_classification', 'itri_nir_flavor_classification', 'ftir_flavor_classification', 
                    'nir_flavor_regression', 'itri_nir_flavor_regression', 'ftir_flavor_regression', 
                    'nir_country_classification', 'itri_nir_country_classification', 'ftir_country_classification', 
                    'nir_process_method_classification', 'itri_nir_process_method_classification',
                    'ftir_process_method_classification']


