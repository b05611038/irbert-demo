"""
Argument configuration for IR-BERT training (public demo version).

Includes options for model architecture, task selection, data path, and optimization settings.
This file does not include any logic tied to IR-BERT’s processor or tokenization scheme.
"""

import os

import torch

from ..utils import Configuration


__all__ = ['TrainingArgument', 'PretrainArgument']


class TrainingArgument(Configuration):
    def __init__(self,
            output_dir,
            logging_dir = None,
            spectrum_wavelengths = None,
            overwrite_output_dir = False,
            dataset_random_sampling = False,
            save_coffee_dataset_split = False,
            dataloader_num_worker = 0,
            dataloader_pin_memory = True,
            dataloader_drop_last = False,
            masked_ratio = 0.,
            progress_unit = 'steps',
            train_steps = 10000,
            record_interval = 10,
            eval_steps = 1000,
            eval_delay = 0,
            save_eval_csv = True,
            test_steps = 1000,
            test_delay = 0,
            save_test_csv = True,
            overwrite_saved_csv = True,
            per_device_batch_size = 8,
            learning_rate = 5e-6,
            weight_decay = 0.001,
            adam_beta1 = 0.9,
            adam_beta2 = 0.999,
            adam_epsilon = 1e-8,
            max_grad_norm = 0.1,
            lr_scheduler_type = 'linear',
            warmup_ratio = 0.1,
            save_steps = 2000,
            start_save_step = 0,
            overwrite_saved_model = True,
            load_best_model_at_end = False,
            seed = 42,
            devices = 'auto',
            ddp_backend = 'auto',
            world_size = 8):

       assert isinstance(output_dir, str)
       assert isinstance(overwrite_output_dir, bool)
       if logging_dir is None:
           logging_dir = os.path.join(output_dir, 'runs')
       else:
           assert isinstance(logging_dir, str)

       if spectrum_wavelengths is not None:
           assert isinstance(spectrum_wavelengths, torch.Tensor)

       assert isinstance(progress_unit, str)
       assert progress_unit in ['steps', 'epochs']
       assert isinstance(dataset_random_sampling, bool)
       assert isinstance(save_coffee_dataset_split, bool)
       assert isinstance(dataloader_num_worker, int)
       assert isinstance(dataloader_pin_memory, bool)
       assert isinstance(dataloader_drop_last, bool)
       assert dataloader_num_worker >= 0
       assert isinstance(masked_ratio, float)
       assert masked_ratio >= 0. and masked_ratio <= 1.

       assert isinstance(train_steps, int)
       assert train_steps >= 0
       assert isinstance(record_interval, int)
       assert record_interval > 0
       assert isinstance(eval_steps, int)
       assert isinstance(eval_delay, int)
       assert eval_delay >= 0
       assert isinstance(save_eval_csv, bool)
       assert isinstance(test_steps, int)
       assert isinstance(test_delay, int)
       assert test_delay >= 0
       assert isinstance(save_test_csv, bool)
       assert isinstance(overwrite_saved_csv, bool)

       assert isinstance(per_device_batch_size, int)
       assert per_device_batch_size > 0
       assert isinstance(learning_rate, float)
       assert isinstance(weight_decay, float)
       assert isinstance(adam_beta1, float)
       assert isinstance(adam_beta2, float)
       assert isinstance(adam_epsilon, float)
       assert isinstance(max_grad_norm, float)
       assert max_grad_norm > 0.
       assert isinstance(lr_scheduler_type, str)
       assert lr_scheduler_type in ['linear', 'cosine_annealing']
       assert isinstance(warmup_ratio, float)
       assert warmup_ratio < 1. and warmup_ratio >= 0.

       assert isinstance(save_steps, int)
       assert isinstance(start_save_step, int)
       assert start_save_step >= 0
       assert isinstance(overwrite_saved_model, bool)
       assert isinstance(load_best_model_at_end, bool)

       assert isinstance(seed, int)
       assert isinstance(devices, str)
       assert isinstance(ddp_backend, str)
       assert ddp_backend in ['auto', 'gloo', 'nccl']
       assert isinstance(world_size, int)

       self.output_dir = output_dir
       self.logging_dir = logging_dir
       self.overwrite_output_dir = overwrite_output_dir

       self.save_coffee_dataset_split = save_coffee_dataset_split
       self.dataset_random_sampling = dataset_random_sampling

       self.dataloader_num_worker = dataloader_num_worker
       self.dataloader_pin_memory = dataloader_pin_memory
       self.dataloader_drop_last = dataloader_drop_last
       self.masked_ratio = masked_ratio

       self.progress_unit = progress_unit
       self.train_steps = train_steps
       self.record_interval = record_interval
       self.eval_steps = eval_steps
       self.eval_delay = eval_delay
       self.save_eval_csv = save_eval_csv
       self.test_steps = test_steps
       self.test_delay = test_delay
       self.save_test_csv = save_test_csv
       self.overwrite_saved_csv = overwrite_saved_csv

       self.per_device_batch_size = per_device_batch_size
       self.learning_rate = learning_rate
       self.weight_decay = weight_decay
       self.adam_beta1 = adam_beta1
       self.adam_beta2 = adam_beta2
       self.adam_epsilon = adam_epsilon
       self.max_grad_norm = max_grad_norm
       self.lr_scheduler_type = lr_scheduler_type
       self.warmup_ratio = warmup_ratio

       self.save_steps = save_steps
       self.start_save_step = start_save_step
       self.overwrite_saved_model = overwrite_saved_model
       self.load_best_model_at_end = load_best_model_at_end

       self.seed = seed
       self.ddp_backend = ddp_backend
       self.world_size = world_size

       super(TrainingArgument, self).__init__(
               output_dir = output_dir,
               logging_dir = logging_dir,
               overwrite_output_dir = overwrite_output_dir,
               save_coffee_dataset_split = save_coffee_dataset_split,
               dataset_random_sampling = dataset_random_sampling,
               dataloader_num_worker = dataloader_num_worker,
               dataloader_pin_memory = dataloader_pin_memory,
               masked_ratio = masked_ratio,
               progress_unit = progress_unit,
               train_steps = train_steps,
               record_interval = record_interval,
               eval_steps = eval_steps,
               eval_delay = eval_delay,
               save_eval_csv = save_eval_csv,
               test_steps = test_steps,
               test_delay = test_delay,
               save_test_csv = save_test_csv,
               per_device_batch_size = per_device_batch_size,
               learning_rate = learning_rate,
               weight_decay = weight_decay,
               adam_beta1 = adam_beta1,
               adam_beta2 = adam_beta2,
               adam_epsilon = adam_epsilon,
               max_grad_norm = max_grad_norm,
               lr_scheduler_type = lr_scheduler_type,
               warmup_ratio = warmup_ratio,
               save_steps = save_steps,
               start_save_step = start_save_step,
               overwrite_saved_model = overwrite_saved_model,
               load_best_model_at_end = load_best_model_at_end,
               seed = seed,
               devices = devices,
               ddp_backend = ddp_backend,
               world_size = world_size)


class PretrainArgument(TrainingArgument):
    def __init__(self,
            output_dir = 'irbert-base',
            logging_dir = None,
            spectrum_wavelengths = None,
            overwrite_output_dir = False,
            save_coffee_dataset_split = True,
            dataset_random_sampling = False,
            dataloader_num_worker = 0,
            dataloader_pin_memory = True,
            dataloader_drop_last = False,
            masked_ratio = 0.1,
            progress_unit = 'steps',
            train_steps = 100000,
            record_interval = 10,
            eval_steps = 1000,
            eval_delay = 0,
            save_eval_csv = True,
            test_steps = 1000,
            test_delay = 0,
            save_test_csv = True,
            overwrite_saved_csv = True,
            per_device_batch_size = 8,
            learning_rate = 5e-7,
            weight_decay = 0.001,
            adam_beta1 = 0.9,
            adam_beta2 = 0.999,
            adam_epsilon = 1e-8,
            max_grad_norm = 0.1,
            lr_scheduler_type = 'linear',
            warmup_ratio = 0.1,
            save_steps = 10000,
            start_save_step = 0,
            overwrite_saved_model = True,
            load_best_model_at_end = False,
            seed = 42,
            devices = 'auto',
            ddp_backend = 'auto',
            world_size = 8):

       super(PretrainArgument, self).__init__(
               output_dir = output_dir,
               logging_dir = logging_dir,
               overwrite_output_dir = overwrite_output_dir,
               save_coffee_dataset_split = save_coffee_dataset_split,
               dataset_random_sampling = dataset_random_sampling,
               dataloader_num_worker = dataloader_num_worker,
               dataloader_pin_memory = dataloader_pin_memory,
               masked_ratio = masked_ratio,
               progress_unit = progress_unit,
               train_steps = train_steps,
               record_interval = record_interval,
               eval_steps = eval_steps,
               eval_delay = eval_delay,
               save_eval_csv = save_eval_csv,
               test_steps = test_steps,
               test_delay = test_delay,
               save_test_csv = save_test_csv,
               overwrite_saved_csv = overwrite_saved_csv,
               per_device_batch_size = per_device_batch_size,
               learning_rate = learning_rate,
               weight_decay = weight_decay,
               adam_beta1 = adam_beta1,
               adam_beta2 = adam_beta2,
               adam_epsilon = adam_epsilon,
               max_grad_norm = max_grad_norm,
               lr_scheduler_type = lr_scheduler_type,
               warmup_ratio = warmup_ratio,
               save_steps = save_steps,
               start_save_step = start_save_step,
               overwrite_saved_model = overwrite_saved_model,
               load_best_model_at_end = load_best_model_at_end,
               seed = seed,
               devices = devices,
               ddp_backend = ddp_backend,
               world_size = world_size)


