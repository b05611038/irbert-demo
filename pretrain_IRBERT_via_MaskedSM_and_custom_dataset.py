import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from IRBert import (IRBertConfig,
                    IRBertForMaskedSM,
                    PretrainArgument,
                    TensorDatasetTrainer)

# your loss function
def my_loss_function(model_outputs, target):
    return criterion(model_outputs, target)

def main():
    train_steps = 100000
    masked_ratio = 0.1
    model_name = 'my-irbert'

    # please load your data to construct your custom dataset
    generated_sample_number = 1024
    wavelengths = [i for i in range(700, 2500, 2)]
    my_spectra = torch.randn(generated_sample_number, len(wavelengths))
    wavelengths = torch.tensor(wavelengths, dtype = torch.float)

    # clone the data for reconstruction
    reconstructed_spectra = my_spectra.clone().detach()

    # Initial the TensorDataset
    dataset = TensorDataset(my_spectra, reconstructed_spectra)

    # define model config (this is the base version.)
    model_config = IRBertConfig(wavelength_window_size = 25.,
                                num_hidden_layer = 12,
                                num_hidden_head = 12,
                                hidden_size = 768,
                                feedforward_size = 3024)

    # defined the pretrained model object
    model = IRBertForMaskedSM(model_config, use_cache = True)
    arguments = PretrainArgument(output_dir = model_name,
                                 train_steps = train_steps,
                                 save_coffee_dataset_split = False,
                                 eval_steps = -1, # no evaluation when it set to -1
                                 test_steps = -1, # no testing when it set to -1
                                 save_steps = -1, # not save model during training when it set to -1
                                 masked_ratio = masked_ratio)

    # if the custom_loss_function is None, it use the same API as training on CoffeeDatabaseDataset.
    # Actually, in the pretraining phase, setting None is recommended.
    trainer = TensorDatasetTrainer(args = arguments,
                                   model = model,
                                   spectrum_wavelengths = wavelengths,
                                   train_dataset = dataset,
                                   train_task_name = 'mask_reconstruction',
                                   train_task = 'regression',
                                   custom_loss_function = my_loss_function)

    trainer.train()

    model = trainer.model
    model.save_pretrained(model_name)

    print('Finish pre-training {0}'.format(model_name))

    return None

if __name__ == '__main__':
    main()

    
