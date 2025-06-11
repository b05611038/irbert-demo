import os
import platform

import abc
import copy
import math
import random
import logging

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from torch.utils.data import (Dataset, 
                              DataLoader,
                              DistributedSampler)

from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

coffee_database_available = True
try:
    import coffee_database
except ImportError:
        coffee_database_available = False

if coffee_database_available:
    from coffee_database.dataset import (CoffeeDatabaseDataset,
                                         EnsembleCoffeeDatabaseDataset)

    from ..data import MultiTaskingEnsembleDataset

from .utils import (default_loss_computing,
                    default_metrics_computing)

from .argument import TrainingArgument
from .optim import create_scheduled_optimizer

from ..utils import (init_directory,
                     clean_directory,
                     setup_distributed_env,
                     cleanup_distributed_env,
                     tcp_port_in_use)

from ..data import (ImplementedTasks,
                    IRSpectrumMaskGenerator)

from ..model import (IRBertConfig,
                     IRBertTokenizer,
                     MultiTaskTokenizer,
                     IRBertProcessor,
                     IRBertForMaskedSM,
                     IRBertForMultiTaskPrediction,
                     IRBertForMaskedSMOutput,
                     IRBertForMultiTaskPredictionOutput)


OutputObject = {IRBertForMaskedSM: IRBertForMaskedSMOutput,
                IRBertForMultiTaskPrediction: IRBertForMultiTaskPredictionOutput}


if coffee_database_available:
    __all__ = ['BaseTrainer', 'CoffeeDatabaseTrainer', 'TensorDatasetTrainer']
else:
    __all__ = ['BaseTrainer', 'TensorDatasetTrainer']


class BaseTrainer(abc.ABC):
    def __init__(self,
            args,
            spectrum_wavelengths = None,
            train_dataset = None,
            train_task_name = None,
            train_task = None,
            eval_dataset = None,
            eval_task_name = None,
            eval_task = None,
            test_dataset = None,
            test_task_name = None,
            test_task = None,
            custom_loss_function = None,
            custom_metric_evaluation = None,
            model = None):

        self.__is_in_train = False
        assert isinstance(args, TrainingArgument)
        self.args = args

        seed = args.get('seed', 42)
        assert isinstance(seed, int)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.seed = seed

        output_dir = args.get('output_dir', 'irbert')
        self.output_dir = output_dir

        overwrite_output_dir = args.get('overwrite_output_dir', False)
        self.overwrite_output_dir = overwrite_output_dir

        logging_dir = args.get('logging_dir', os.path.join(output_dir, 'runs'))
        self.logging_dir = logging_dir

        if spectrum_wavelengths is None:
            spectrum_wavelengths = args.get('spectrum_wavelengths', None)
            if spectrum_wavelengths is None:
                raise ValueError('Need to set wavelengths when initializing trainer or in TrainingArgument.')
        else:
            assert isinstance(spectrum_wavelengths, (torch.Tensor, dict))

        self.spectrum_wavelengths = spectrum_wavelengths

        if custom_loss_function is not None:
            assert callable(custom_loss_function)

        self.custom_loss_function = custom_loss_function
        if custom_metric_evaluation is not None:
            assert callable(custom_metric_evaluation)

        self.custom_metric_evaluation = custom_metric_evaluation
        default_computing = True
        if (self.custom_loss_function is not None) and (self.custom_metric_evaluation is not None):
            default_computing = False

        available_tasks = ['auto', 'regression', 'multiclass_classification', 'multilabel_classification']
        coffee_task_names = ImplementedTasks
        if train_dataset is not None:
            assert isinstance(train_dataset, Dataset)
            if default_computing:
                if train_task is None:
                    raise ValueError('Argument: train_task cannot be None when using default computing mode.')

                assert train_task in available_tasks
                if train_task != 'auto':
                    assert train_task_name is not None
                    assert isinstance(train_task_name, str)
                    assert len(train_task_name) > 0

        if eval_dataset is not None:
            assert isinstance(eval_dataset, Dataset)
            if default_computing:
                if eval_task is None:
                    raise ValueError('Argument: eval_task cannot be None when using default computing mode.')

                assert eval_task in available_tasks
                if eval_task != 'auto':
                    assert eval_task_name is not None
                    assert isinstance(eval_task_name, str)
                    assert len(eval_task_name) > 0

        if test_dataset is not None:
            assert isinstance(test_dataset, Dataset)
            if default_computing:
                if test_task is None:
                    raise ValueError('Argument: test_task cannot be None when using default computing mode.')

                assert test_task in available_tasks
                if test_task != 'auto':
                    assert test_task_name is not None
                    assert isinstance(test_task_name, str)
                    assert len(test_task_name) > 0

        save_coffee_dataset_split = args.get('save_coffee_dataset_split', False)
        if save_coffee_dataset_split:
            if not coffee_database_available:
                raise ValueError('Becuase coffee_database is not installed, save_coffee_dataset_split' + \
                        ' cannot be set to True.')

            save_split_error_flag = False
            if train_dataset is not None:
                if not isinstance(train_dataset, (CoffeeDatabaseDataset, EnsembleCoffeeDatabaseDataset)):
                    save_split_error_flag = True

            if eval_dataset is not None:
                if not isinstance(eval_dataset, (CoffeeDatabaseDataset, EnsembleCoffeeDatabaseDataset)):
                    save_split_error_flag = True

            if test_dataset is not None:
                if not isinstance(test_dataset, (CoffeeDatabaseDataset, EnsembleCoffeeDatabaseDataset)):
                    save_split_error_flag = True

            if save_split_error_flag:
                raise ValueError('Because inputted dataset is not a coffee database dataset,' + \
                        ' save_split_error_flag cannot be True.')

        pretraining = False
        if train_task_name is not None:
            if 'mask_reconstruction' in train_task_name:
                pretraining = True

        self.train_from_scratch = False
        if model is None:
            self.train_from_scratch = True
            model_config = IRBertConfig()
            if pretraining:
                model = IRBertForMaskedSM(model_config)
            else:
                model = IRBertForMultiTaskPrediction(model_config)

            self.model = model
        else:
            assert isinstance(model, (IRBertForMaskedSM,
                                      IRBertForMultiTaskPrediction))

            self.model = model

        self.tokenizer = self.model.tokenizer
        self.task_prediction = True
        if isinstance(self.model, IRBertForMaskedSM):
            self.task_prediction = False

        self.save_coffee_dataset_split = save_coffee_dataset_split

        self.train_dataset = train_dataset
        self.train_task = train_task
        self.train_task_name = train_task_name
        self.eval_dataset = eval_dataset
        self.eval_task = eval_task
        self.eval_task_name = eval_task_name
        self.test_dataset = test_dataset
        self.test_task = test_task
        self.test_task_name = test_task_name

        dataset_random_sampling = args.get('dataset_random_sampling', False)
        self.dataset_random_sampling = dataset_random_sampling

        dataloader_num_worker = args.get('dataloader_num_worker', 0)
        assert dataloader_num_worker >= 0
        self.dataloader_num_worker = dataloader_num_worker

        dataloader_pin_memory = args.get('dataloader_pin_memory', True)
        assert isinstance(dataloader_pin_memory, bool)
        self.dataloader_pin_memory = dataloader_pin_memory

        dataloader_drop_last = args.get('dataloader_drop_last', False)
        assert isinstance(dataloader_drop_last, bool)
        self.dataloader_drop_last = dataloader_drop_last

        masked_ratio = args.get('masked_ratio', 0.)
        self.masked_ratio = masked_ratio
        if masked_ratio > 0.:
            self.spectrum_mask_generator = IRSpectrumMaskGenerator(masked_ratio = masked_ratio)
        else:
            self.spectrum_mask_generator = None

        progress_unit = args.get('progress_unit', 'steps')
        if progress_unit not in ['steps']:
            raise NotImplementedError("args.progress_unit='{0}' is not implemented."\
                    .format(progress_unit))

        self.progress_unit = progress_unit

        train_steps = args.get('train_steps', 10000)
        self.train_steps = train_steps
        record_interval = args.get('record_interval', 10)
        self.record_interval = record_interval

        eval_steps = args.get('eval_steps', 1000)
        if eval_steps >= 1:
            assert eval_steps <= train_steps
        else:
            eval_steps = train_steps + 1

        self.eval_steps = eval_steps

        eval_delay = args.get('eval_delay', 0)
        assert eval_delay < train_steps
        self.eval_delay = eval_delay
        save_eval_csv = args.get('save_eval_csv', False)
        self.save_eval_csv = save_eval_csv

        test_steps = args.get('test_steps', 1000)
        if test_steps >= 1:
            assert test_steps <= train_steps
        else:
            test_steps = train_steps + 1

        self.test_steps = test_steps
        test_delay = args.get('test_delay', 0)
        assert eval_delay < train_steps
        self.test_delay = test_delay
        save_test_csv = args.get('save_test_csv', False)
        self.save_test_csv = save_test_csv

        overwrite_saved_csv = args.get('overwrite_saved_csv', True)
        self.overwrite_saved_csv = overwrite_saved_csv

        per_device_batch_size = args.get('per_device_batch_size', 8)
        assert per_device_batch_size >= 1
        self.per_device_batch_size = per_device_batch_size

        save_steps = args.get('save_steps', 10000)
        if save_steps >= 1:
            assert save_steps <= train_steps
        else:
            save_steps = train_steps + 1

        self.save_steps = save_steps

        start_save_step = args.get('start_save_step', 0)
        assert start_save_step >= 0
        self.start_save_step = start_save_step

        overwrite_saved_model = args.get('overwrite_saved_model', True)
        assert isinstance(overwrite_saved_model, bool)
        self.overwrite_saved_model = overwrite_saved_model

        load_best_model_at_end = args.get('load_best_model_at_end', False)
        assert isinstance(load_best_model_at_end, bool)
        self.load_best_model_at_end = load_best_model_at_end

        devices = args.get('devices', 'auto')
        if devices == 'auto':
            if torch.cuda.is_available():
                self.devices = [i for i in range(torch.cuda.device_count())]
            else:
                self.devices = None

        else:
            if devices == 'cpu':
                self.devices = None
            elif 'cuda' in devices: # assign specific GPU ('cuda:0', 'cuda:1')
                self.devices = [int(devices.replace('cuda', '').split(':')[-1])]
            else:
                raise ValueError('Device: {0} is not valid in {1}'.format(devices,
                        self.__class__.__name__))

        ddp_backend = args.get('ddp_backend', 'auto')
        assert ddp_backend in ['auto', 'nccl', 'gloo']
        if ddp_backend == 'auto':
            if torch.cuda.is_available():
                ddp_backend = 'nccl'
            else:
                ddp_backend = 'gloo'

        elif ddp_backend == 'nccl':
            if not torch.cuda.is_available():
                ddp_backend = 'gloo'
                self.devices = None

        self.ddp_backend = ddp_backend
        if ddp_backend == 'gloo':
            self.dataloader_pin_memory = False 

        world_size = args.get('world_size', 8)
        if ddp_backend == 'nccl':
            world_size = len(self.devices)

        self.world_size = world_size

        self.writer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.model_wrapped = None
        self.recorded_best_loss = float('inf')

        self.eval_metrics = None
        self.eval_reports = None
        self.test_metrics = None
        self.test_reports = None

        self.__finish_training = False

    def __repr__(self):
        return '{0}(\n'.format(self.__class__.__name__) + \
                'args={1}\n'.format(self.args) + ')'

    @property
    def finish_training(self):
        return self.__finish_training

    @property
    def is_in_train(self):
        return self.__is_in_train

    def setup_dist_env(self, host = 'localhost', port = None):
        system = platform.system()
        if system == 'Linux' or system == 'Darwin':
            if port is None:
                port_in_use = True
                while port_in_use:
                    port = random.randrange(1001, 32768)
                    if not tcp_port_in_use(host, port):
                        port_in_use = False

                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = str(port)
            else:
                if tcp_port_in_use(host, port):
                    raise RuntimeError('{0}:{1} is in use, please set another port'\
                            .format(host, port))

        return None

    def setup_ddp(self, rank, world_size):
        setup_distributed_env(rank, 
                              world_size,
                              backend = self.ddp_backend)

        if self.ddp_backend == 'nccl':
            device_ids = [rank]
            self.model = self.model.to(rank)
        else:
            device_ids = None

        self.model = DDP(self.model, 
                         device_ids = device_ids)

        self.model._set_static_graph()
        self.model_wrapped = self.model.module

        return None

    def clean_ddp(self, rank):
        if isinstance(self.model_wrapped, DDP):
            self.model = self.model_wrapped.module
            self.model_wrapped = None

        cleanup_distributed_env()

        if self.writer is not None:
            self.writer.close()
            self.writer = None

        return None

    def setup_logger(self, mode, rank):
        logging_dir = self.logging_dir
        if rank == 0:
            logging_dir = init_directory(logging_dir)

        logger = logging.getLogger('{0}Logger'.format(mode))
        logger.setLevel(logging.DEBUG)

        # handle logfile
        if rank == 0:
            log_file_name = os.path.split(self.output_dir)[-1]
            log_file_path = os.path.join(logging_dir, '{0}.log'.format(log_file_name))

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)

            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
        else:
            file_handler = None

        # handle screen output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        if file_handler is not None:
            logger.addHandler(file_handler)

        logger.addHandler(console_handler)

        self.logger = logger

        return None

    def _create_dataloader(self, rank, world_size, dataset, mode, args = None):
        if args is None:
            args = self.args

        assert isinstance(mode, str)
        assert mode in ['train', 'eval']

        if coffee_database_available:
            raise_error = False
            if self.dataset_random_sampling:
                if not isinstance(dataset, EnsembleCoffeeDatabaseDataset):
                    raise_error = True
                else:
                    if isinstance(dataset, MultiTaskingEnsembleDataset):
                        raise_error = True

                if raise_error:
                    raise NotImplementedError('Please set dataset_random_sampling=False,' + \
                            ' the dataset not implemented with the random_sampling function.')
        else:                        
            if self.dataset_random_sampling:
                raise ValueError('Argument: dataset_random_sampling is designed for dataset in' + \
                       ' private library: coffee_database, set it to False if using the custom dataset.')

        sampler = DistributedSampler(dataset, 
                                     num_replicas = self.world_size, 
                                     rank = rank)

        pin_memory_device = ''
        if self.ddp_backend == 'nccl':
            pin_memory_device = 'cuda'

        return DataLoader(dataset, 
                          batch_size = self.per_device_batch_size,
                          sampler = sampler,
                          num_workers = self.dataloader_num_worker,
                          pin_memory = self.dataloader_pin_memory,
                          drop_last = self.dataloader_drop_last,
                          pin_memory_device = pin_memory_device)

    def _all_reduce_tensor(self, tensor, rank):
        if self.devices is not None:
            tensor = tensor.to(rank)
        
        dist.all_reduce(tensor, op = dist.ReduceOp.SUM)
        if self.devices is not None:
            torch.cuda.synchronize(rank)

        tensor /= dist.get_world_size()

        return tensor

    def train(self, 
            train_dataset = None, 
            train_task_name = None,
            train_task = None,
            eval_dataset = None,
            eval_task_name = None,
            eval_task = None,
            test_dataset = None,
            test_task_name = None,
            test_task = None,
            custom_loss_function = None,
            custom_metric_evaluation = None,
            world_size = None):

        if world_size is None:
            world_size = self.world_size
        else:
            if torch.cuda.is_available():
                if self.ddp_backend == 'nccl':
                    if world_size > torch.cuda.device_count():
                        raise ValueError('Cannot set world_size more than' + \
                                ' the number of CUDA devices.')

        if custom_loss_function is not None:
            assert callable(custom_loss_function)
            self.custom_loss_function = custom_loss_function

        if custom_metric_evaluation is not None:
            assert callable(custom_metric_evaluation)
            self.custom_metric_evaluation = custom_metric_evaluation

        default_computing = True
        if (self.custom_loss_function is not None) and (self.custom_metric_evaluation is not None):
            default_computing = False

        self.__is_in_train = True
        self.output_dir = init_directory(self.output_dir)
        if self.overwrite_output_dir:
            clean_directory(self.output_dir)
        else:
            if os.path.isdir(self.output_dir):
                if len(os.listdir(self.output_dir)) > 0:
                    message = 'Output_dir: {0} is not empty, '.format(self.output_dir)
                    message += 'set overwrite_output_dir=True to force program run and overwrite old files.'
                    raise RuntimeError(message)

        if train_dataset is None:
            train_dataset = self.train_dataset
        else:
            assert isinstance(train_dataset, Dataset)

        if train_task_name is None:
            train_task_name = self.train_task_name

        if train_task is None:
            train_task = self.train_task

        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        else:
            assert isinstance(eval_dataset, Dataset)

        if eval_task_name is None:
            eval_task_name = self.eval_task_name

        if eval_task is None:
            eval_task = self.eval_task

        if eval_dataset is not None:
            if default_computing:
                assert eval_task_name is not None
                assert eval_task is not None

        if test_dataset is None:
            test_dataset = self.test_dataset
        else:
            assert isinstance(test_dataset, Dataset)

        if test_task_name is None:
            test_task_name = self.test_task_name

        if test_task is None:
            test_task = self.test_task

        if test_dataset is not None:
            if default_computing:
                assert test_task_name is not None
                assert test_task is not None

        if coffee_database_available:
            if self.dataset_random_sampling:
                random_sampling_error_flag = False
                if not isinstance(train_dataset, EnsembleCoffeeDatabaseDataset):
                    random_sampling_error_flag = True
                else:
                    if isinstance(train_dataset, MultiTaskingEnsembleDataset):
                        random_sampling_error_flag = True

                if random_sampling_error_flag:
                    raise NotImplementedError('Please set dataset_random_sampling=False,' + \
                            ' other dataset was not implemented the random_sampling.')
        else:
            if self.dataset_random_sampling:
                raise ValueError('Argument: dataset_random_sampling is designed for dataset in' + \
                       ' private library: coffee_database, set it to False if using the custom dataset.')

        self.setup_dist_env()
        mp.set_start_method('spawn', force = True)
        self.model = self.model.cpu()

        manager = mp.Manager()
        model_container = manager.dict()

        mp.spawn(self._distributed_train,
                 args = (world_size, model_container, 
                         train_dataset, train_task_name, train_task,
                         eval_dataset, eval_task_name, eval_task, 
                         test_dataset, test_task_name, test_task), 
                 nprocs = world_size, 
                 join = True)

        if 'state_dict' in model_container:
            self.model = self.model.cpu()
            self.model.load_state_dict(model_container['state_dict'])
        else:
            raise RuntimeError('Cannot fetch trained model parameter from rank 0 process.')

        if self.save_coffee_dataset_split:
            if isinstance(train_dataset, (CoffeeDatabaseDataset, EnsembleCoffeeDatabaseDataset)):
                split_filename = os.path.join(self.output_dir, 'dataset_split.json')
                train_dataset.save_split(fname = split_filename, val = True, force_save = True)

        if self.load_best_model_at_end:
            filename = 'model_best'
            self.model = self.model.from_pretrained(self.output_dir, filename = filename)

        if dist.is_initialized():
            dist.destroy_process_group()

        self.__finish_training = True
        self.__is_in_train = False

        return None

    def _distributed_train(self, 
            rank, 
            world_size, 
            model_container,
            train_dataset,
            train_task_name,
            train_task,
            eval_dataset, 
            eval_task_name,
            eval_task,
            test_dataset,
            test_task_name,
            test_task):

        is_coffee_dataset = False
        if coffee_database_available:
            if isinstance(train_dataset, (CoffeeDatabaseDataset, EnsembleCoffeeDatabaseDataset)):
                is_coffee_dataset = True

        is_multi_task, multi_task_together = False, False
        if coffee_database_available:
            if isinstance(train_dataset, EnsembleCoffeeDatabaseDataset):
                is_multi_task = True
                if self.dataset_random_sampling:
                    train_dataset.random_sampling = True

                if isinstance(train_dataset, MultiTaskingEnsembleDataset):
                    multi_task_together = True

        overwrite_saved_csv = self.overwrite_saved_csv
        if self.custom_metric_evaluation is None:
            save_eval_csv = self.save_eval_csv
            save_test_csv = self.save_test_csv
        else:
            save_eval_csv = False
            save_test_csv = False

        self.setup_ddp(rank, world_size)
        self.setup_logger('Train', rank)
        if rank == 0:
            self.logger.info('Successfully setup the environment of distributed data parallel (DDP) training.')
            self.logger.info('Start distributed training in {0} ...'\
                    .format(self.__class__.__name__))
            self.logger.info('Produced model files all save at {0}'.format(self.output_dir))
            self.logger.info('Produced logging files all save at {0}'.format(self.logging_dir))

            if self.train_from_scratch:
                self.logger.info('Default model: {0} initialized by trainer'.format(type(self.model)))

        dataset_length = len(train_dataset)
        if is_coffee_dataset:
            train_dataset.set_mode('train')
            if eval_dataset is not None:
                eval_dataset.set_mode('val')

            if test_dataset is not None:
                test_dataset.set_mode('test')

        dataloader = self._create_dataloader(rank, world_size, train_dataset, 'train', self.args)
        if rank == 0:
            dataset_name = train_dataset.__class__.__name__
            if self.model_wrapped is not None:
                model_name = self.model_wrapped.__class__.__name__
            else:
                model_name = self.model.__class__.__name__
            
            self.logger.info('Using {0} for training the {1} model'.format(dataset_name, model_name))

        train_steps = self.train_steps
        record_interval = self.record_interval

        eval_steps = self.eval_steps
        eval_delay = self.eval_delay
        test_steps = self.test_steps
        test_delay = self.test_delay

        save_steps = self.save_steps
        start_save_step = self.start_save_step
        overwrite_saved_model = self.overwrite_saved_model

        total_epoch = math.ceil(train_steps / len(dataloader))
        if rank == 0:
            self.writer = SummaryWriter(log_dir = self.logging_dir)
            progress_bar = tqdm(total = train_steps,
                                desc = 'Training Progress', 
                                unit = 'it')
        else:
            progress_bar = None

        (optimizer,
         lr_scheduler) = create_scheduled_optimizer(self.args, model = self.model)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        max_grad_norm = self.args.max_grad_norm

        self.model = self.model.train()
        step_now, keep_training, eval_flag = 0, True, False
        for epoch in range(total_epoch):
            dataloader.sampler.set_epoch(epoch)
            for _, packed_data in enumerate(dataloader):
                if multi_task_together:
                    spectrum = packed_data[0]
                    multi_targets = packed_data[1: ]
                else:
                    spectrum, target = packed_data

                train_cache_cleaning = False

                task_name = train_task_name
                task_type = train_task
                if is_coffee_dataset:
                    if train_task == 'auto':
                        task_name = self.train_dataset.task_now
                        if multi_task_together:
                            task_types = {}
                            for tname in task_name:
                                task_types[tname] = self.tokenizer.get_task_type(tname) 
                        else:
                            task_type = self.tokenizer.get_task_type(task_name)

                if is_multi_task:
                    if multi_task_together:
                        spectrum_wavelengths = self.spectrum_wavelengths
                    else:
                        spectrum_wavelengths = self.spectrum_wavelengths[task_name]
                else:
                    spectrum_wavelengths = self.spectrum_wavelengths

                spectrum_wavelengths = spectrum_wavelengths.clone().detach()
                if self.ddp_backend == 'nccl':
                    spectrum = spectrum.to(self.devices[rank])
                    spectrum_wavelengths = spectrum_wavelengths.to(self.devices[rank])
                    if multi_task_together:
                        for target_idx in range(len(multi_targets)):
                            multi_targets[target_idx] = multi_targets[target_idx].to(self.devices[rank])
                    else:
                        target = target.to(self.devices[rank])

                spectrum_mask = None
                if self.spectrum_mask_generator is not None:
                    spectrum_mask = self.spectrum_mask_generator(spectrum)

                masking_prediction = False
                if task_name is not None:
                    if isinstance(task_name, (tuple, list)):
                        targeted_tasks = task_name
                    else:
                        if 'reconstruction' in task_name:
                            targeted_tasks = None
                            if 'mask' in task_name:
                                masking_prediction = True
                        else:          
                            targeted_tasks = (task_name, )
                else:
                    targeted_tasks = self.tokenizer.tasks

                optimizer.zero_grad()
                if self.task_prediction:
                    # Model call redacted in demo version
                    outputs = self.model(spectrum = spectrum,
                                         spectrum_wavelengths = spectrum_wavelengths,
                                         targeted_tasks = targeted_tasks)
                else:
                    # Model call redacted in demo version
                    outputs = self.model(spectrum = spectrum,
                                         spectrum_wavelengths = spectrum_wavelengths)

                if masking_prediction:
                    if spectrum_mask is not None:
                        target = target[:, spectrum_mask]

                if multi_task_together:
                    loss_value = torch.tensor(0., dtype = spectrum.dtype, device = spectrum.device)
                    for task_idx in range(len(task_name)):
                        tname = task_name[task_idx]
                        task_type = task_types[tname]
                        single_task_loss = self.compute_loss(outputs, multi_targets[task_idx], tname, task_type)

                        weighting = outputs['tasks_weighting'][tname]
                        if 'classification' in task_type:
                            loss_value += ((single_task_loss / (2. * torch.exp(weighting))) + torch.sqrt(torch.exp(weighting)))
                        else:
                            loss_value += ((single_task_loss / torch.exp(weighting)) + torch.sqrt(torch.exp(weighting)))
                else:
                    loss_value = self.compute_loss(outputs, target, task_name, task_type)

                loss_value.backward()

                if (step_now + 1) % record_interval == 0:
                    has_training_record = True
                    grad_norm = self.compute_gradient_norm()
                    reduced_grad_norm = self._all_reduce_tensor(grad_norm, rank).item()
                else:
                    has_training_record = False

                clip_grad_norm_(self.model.parameters(), max_norm = max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                torch.distributed.barrier()

                step_now += 1

                if has_training_record:
                    reduced_loss = self._all_reduce_tensor(loss_value.detach().clone(), rank).item()
                    learning_rate = self.get_learning_rate()

                if rank == 0:
                    progress_bar.update(1)
                    if has_training_record:
                        layout_dict = {'lr': learning_rate,  
                                       'loss': reduced_loss, 
                                       'grad_norm': reduced_grad_norm,
                                       'iter': step_now,}

                        tqdm.write(str(layout_dict))
                        self.writer.add_scalar('learning_rate', learning_rate, step_now)
                        self.writer.add_scalar('train/loss', reduced_loss, step_now)
                        if not math.isnan(reduced_grad_norm):
                            self.writer.add_scalar('train/grad_norm', reduced_grad_norm, step_now)

                if step_now >= eval_delay:
                    if ((step_now - eval_delay) % eval_steps == 0) or (step_now == train_steps):
                        if (eval_dataset is not None) and (len(eval_dataset) > 0):
                            self._distributed_force_sync_model(rank, world_size)
                            if not train_cache_cleaning:
                                del spectrum, spectrum_wavelengths, spectrum_mask
                                del outputs
                                del loss_value
                                if multi_task_together:
                                    del multi_targets
                                else:
                                    del target

                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                train_cache_cleaning = True

                            self._distributed_eval(rank, world_size, eval_dataset, eval_task_name, 
                                    eval_task, env_init = False)

                            if rank == 0:
                                summed_eval_loss, has_loss_value = 0., False
                                if is_multi_task:
                                    for task in self.eval_metrics:
                                        for m in self.eval_metrics[task]:
                                            if m == 'loss':
                                                summed_eval_loss += self.eval_metrics[task][m]
                                                has_loss_value = True
                                                continue

                                            self.writer.add_scalar('eval/{0}/{1}'.format(task, m), 
                                                    self.eval_metrics[task][m], step_now)

                                    if has_loss_value:
                                        self.writer.add_scalar('eval/summed_loss', summed_eval_loss, step_now)

                                    if save_eval_csv:
                                        for task in self.eval_reports:
                                            if overwrite_saved_csv:
                                                report_filename = 'eval.{0}.csv'.format(task)
                                            else:
                                                report_filename = 'eval.{0}_it{1}.csv'.format(task, step_now)

                                            report_filename = os.path.join(self.output_dir, report_filename)
                                            if self.eval_reports[task] is not None:
                                                self.eval_reports[task].save_as_csv(report_filename)
                                else:
                                    for m in self.eval_metrics:
                                        self.writer.add_scalar('eval/{0}'.format(m), self.eval_metrics[m], 
                                                step_now)

                                        if m == 'loss':
                                            summed_eval_loss += self.eval_metrics[m]
                                            has_loss_value = True

                                    if save_eval_csv:
                                        if overwrite_saved_csv:
                                            report_filename = 'eval.{0}.csv'.format(eval_task_name)
                                        else:
                                            report_filename = 'eval.{0}_it{1}.csv'.format(eval_task_name, step_now)

                                        report_filename = os.path.join(self.output_dir, report_filename)
                                        if self.eval_reports is not None:
                                            self.eval_reports.save_as_csv(report_filename)

                                if has_loss_value:
                                    if summed_eval_loss <= self.recorded_best_loss:
                                        self.save_model(extra_name = 'best', rank = rank)
                                        self.recorded_best_loss = summed_eval_loss

                                self.eval_reports = None
                                self.eval_metrics = None

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            self.model = self.model.train()

                if step_now >= test_delay:
                    if ((step_now - test_delay) % test_steps == 0) or (step_now == train_steps):
                        if (test_dataset is not None) and (len(test_dataset) > 0):
                            self._distributed_force_sync_model(rank, world_size)
                            if not train_cache_cleaning:
                                del spectrum, spectrum_wavelengths, spectrum_mask
                                del outputs
                                del loss_value
                                if multi_task_together:
                                    del multi_targets
                                else:
                                    del target

                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                train_cache_cleaning = True

                            self._distributed_test(rank, world_size, test_dataset, test_task_name,
                                    test_task, env_init = False)

                            if rank == 0:
                                if is_multi_task:
                                    summed_test_loss, has_loss_value = 0., False
                                    for task in self.test_metrics:
                                        for m in self.test_metrics[task]:
                                            if m == 'loss':
                                                summed_test_loss += self.test_metrics[task][m]
                                                has_loss_value = True
                                                continue

                                            self.writer.add_scalar('test/{0}/{1}'.format(task, m), 
                                                    self.test_metrics[task][m], step_now)

                                    if save_test_csv:
                                        for task in self.test_reports:
                                            if overwrite_saved_csv:
                                                report_filename = 'test.{0}.csv'.format(task)
                                            else:
                                                report_filename = 'test.{0}_it{1}.csv'.format(task, step_now)

                                            report_filename = os.path.join(self.output_dir, report_filename)
                                            if self.test_reports[task] is not None:
                                                self.test_reports[task].save_as_csv(report_filename)
                                   
                                    if has_loss_value:
                                        self.writer.add_scalar('test/summed_loss', summed_test_loss, step_now)
                                else:
                                    for m in self.test_metrics:
                                        self.writer.add_scalar('test/{0}'.format(m), self.test_metrics[m], 
                                                step_now)

                                    if save_test_csv:
                                        if overwrite_saved_csv:
                                            report_filename = 'test.{0}.csv'.format(test_task_name)
                                        else:
                                            report_filename = 'test.{0}_it{1}.csv'.format(test_task_name, step_now)

                                        report_filename = os.path.join(self.output_dir, report_filename)
                                        if self.test_reports is not None:
                                            self.test_reports.save_as_csv(report_filename)

                                self.test_reports = None
                                self.test_metrics = None

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            self.model = self.model.train()

                if step_now >= start_save_step:
                    if (step_now - start_save_step) % save_steps == 0:
                        self._distributed_force_sync_model(rank, world_size)
                        if rank == 0:
                            if overwrite_saved_model:
                                extra_name = ''
                            else:
                                extra_name = 'it{0}'.format(step_now)

                            self.save_model(extra_name = extra_name, rank = rank)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                if step_now >= train_steps:
                    break
                    keep_training = False

                if self.dataset_random_sampling and is_multi_task:
                    train_dataset.random_switch_task()

            if not keep_training:
                break

        self._distributed_force_sync_model(rank, world_size)
        if rank == 0:
            progress_bar.close()
            if isinstance(self.model, DDP):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            model_container['state_dict'] = {k: v.cpu() for k, v in state_dict.items()}

        self.logger = None
        self.clean_ddp(rank)

        return None

    def _distributed_force_sync_model(self, rank, world_size):
        for param in self.model.parameters():
            torch.distributed.all_reduce(param.data, op = torch.distributed.ReduceOp.SUM)
            param.data /= world_size

        return None

    def evaluate(self, 
            eval_dataset = None, 
            eval_task_name = None,
            eval_task = None,
            custom_loss_function = None,
            custom_metric_evaluation = None,
            world_size = None):

        if custom_loss_function is not None:
            assert callable(custom_loss_function)
            self.custom_loss_function = custom_loss_function

        if custom_metric_evaluation is not None:
            assert callable(custom_metric_evaluation)
            self.custom_metric_evaluation = custom_metric_evaluation

        default_computing = True
        if (self.custom_loss_function is not None) and (self.custom_metric_evaluation is not None):
            default_computing = False

        if world_size is None:
            world_size = self.world_size
        else:
            if torch.cuda.is_available():
                if self.ddp_backend == 'nccl':
                    if world_size > torch.cuda.device_count():
                        raise ValueError('Cannot set world_size more than' + \
                                ' the number of CUDA devices.')

        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        else:
            assert isinstance(eval_dataset, Dataset)

        if eval_task_name is None:
            eval_task_name = self.eval_task_name

        if eval_task is None:
            eval_task = self.eval_task

        if not custom_computing:
            assert eval_task_name is not None
            assert eval_task is not None

        if self.dataset_random_sampling:
            if coffee_database_available:
                raise_error = False
                if not isinstance(eval_dataset, EnsembleCoffeeDatabaseDataset):
                    raise_error = True
                else:
                    if isinstance(eval_dataset, MultiTaskingEnsembleDataset):
                        raise_error = True

                if raise_error:
                    raise NotImplementedError('Please set dataset_random_sampling=False,' + \
                            ' other dataset was not implemented the random_sampling.')
            else:
                raise ValueError('Argument: dataset_random_sampling is designed for dataset in' + \
                       ' private library: coffee_database, set it to False if using the custom dataset.')

        self.setup_dist_env()
        mp.spawn(self._distributed_eval,
                 args = (world_size, eval_dataset, eval_task_name, eval_task, 'eval', True),
                 nprocs = world_size, 
                 join = True)

        self.logger = None

        return None

    def _gather_output_dict(self, output_dict_list, rank):
        gathered_output = None
        if len(output_dict_list) > 0:
            keys = output_dict_list[0].keys()
            world_size = dist.get_world_size()

            gathered_output = {key: [] for key in keys}
            for key in keys:
                tensor = torch.cat([d[key] for d in output_dict_list], dim = 0).detach().clone()

                gathered_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(gathered_tensors, tensor)

                if rank == 0:
                    gathered_output[key] = torch.cat(gathered_tensors, dim = 0)

        return gathered_output if rank == 0 else None

    def _gather_target_tensors(self, target_list, rank):
        gathered_target = None
        if len(target_list) > 0:
            target_tensor = torch.cat(target_list, dim = 0).detach().clone()

            gathered_target = [torch.empty_like(target_tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered_target, target_tensor)
            if rank == 0:
                gathered_target = torch.cat(gathered_target, dim = 0)

        return gathered_target if rank == 0 else None

    def _distributed_eval(self, 
            rank, 
            world_size, 
            eval_dataset, 
            eval_task_name,
            eval_task, 
            display_folder = 'eval',
            env_init = True):

        if env_init:
            if display_folder == 'eval':
                logger_name = 'Eval'
            elif display_folder == 'test':
                logger_name = 'Test'
            else:
                logger_name = display_folder
                
            self.setup_logger(logger_name, rank)
            self.setup_ddp(rank, world_size)
            if rank == 0:
                self.writer = SummaryWriter(log_dir = self.logging_dir)
                self.logger.info('Successfully setup the environment of distributed data parallel (DDP) evaluation.')
                self.logger.info('Start distributed evaluation in {0} ...'\
                        .format(self.__class__.__name__))
                self.logger.info('Produced logging files all save at {0}'.format(self.logging_dir))

        is_multi_task, multi_task_together = False, False
        if coffee_database_available: 
            if isinstance(eval_dataset, EnsembleCoffeeDatabaseDataset):
                is_multi_task = True
                if isinstance(eval_dataset, MultiTaskingEnsembleDataset):
                    multi_task_together = True

        self.model = self.model.eval()
        if is_multi_task:
            if multi_task_together:
                tasks = eval_dataset.task_now
            else:
                tasks = eval_dataset.tasks.copy()
        else:
            if self.eval_task_name is None:
                tasks = self.tokenizer.tasks
            else:
                tasks = [self.eval_task_name]

        if rank == 0:
            total_data_number = len(eval_dataset)
            if coffee_database_available:
                if is_multi_task:
                    if not multi_task_together:
                        total_data_number = 0
                        for task in eval_dataset.tasks:
                            total_data_number += eval_dataset.data_number[task][eval_dataset.mode]

            total_batch_size = self.per_device_batch_size * world_size
            total_iteration = math.floor(total_data_number / total_batch_size)
            if total_iteration == 0:
                total_iteration += 1

            if multi_task_together:
                total_iteration += 1
            else:
                total_iteration += len(tasks)

            eval_progress_bar = tqdm(total = total_iteration,
                                     desc = 'Evaluating progress ({0})'.format(display_folder),
                                     unit = 'it',
                                     leave = False)

        with torch.no_grad():
            task_metrics, task_reports = None, None
            if rank == 0:
                task_metrics, task_reports = {}, {}

            for task_now in tasks:
                if is_multi_task:
                    if not multi_task_together:
                        eval_dataset.set_task(task_now)

                dataloader = self._create_dataloader(rank, world_size, eval_dataset, 'eval', self.args)
                reduced_outputs = ['combined_embeddings',
                                   'last_hidden_state',
                                   'last_attention',
                                   'hidden_states',
                                   'attentions',
                                   'tasks_weighting']

                total_loss, total_iter = 0., 0 
                all_outputs = []
                if multi_task_together:
                    all_targets = {}
                    for tname in tasks:
                        all_targets[tname] = []
                else:
                    all_targets = []

                for _, packed_data in enumerate(dataloader):
                    if multi_task_together:
                        spectrum = packed_data[0]
                        multi_targets = packed_data[1: ]
                    else:
                        spectrum, target = packed_data

                    if is_multi_task:
                        if multi_task_together:
                            spectrum_wavelengths = self.spectrum_wavelengths
                        else:
                            spectrum_wavelengths = self.spectrum_wavelengths[task_now]
                    else:
                        spectrum_wavelengths = self.spectrum_wavelengths

                    spectrum_wavelengths = spectrum_wavelengths.detach().clone()
                    if self.ddp_backend == 'nccl':
                        spectrum = spectrum.to(self.devices[rank])
                        spectrum_wavelengths = spectrum_wavelengths.to(self.devices[rank])
                        if multi_task_together:
                            for task_idx in range(len(multi_targets)):
                                multi_targets[task_idx] = multi_targets[task_idx].to(self.devices[rank])
                        else:
                            target = target.to(self.devices[rank])

                    spectrum_mask = None
                    if self.spectrum_mask_generator is not None:
                        spectrum_mask = self.spectrum_mask_generator(spectrum)
                        if 'reconstruction' not in task_now:
                            spectrum_mask = None

                    masking_prediction = False
                    if 'reconstruction' in task_now:
                        targeted_tasks = None
                        if 'mask' in task_now:
                            masking_prediction = True
                    else:
                        targeted_tasks = (task_now, )

                    if multi_task_together:
                        targeted_tasks = tasks

                    if self.task_prediction:
                        outputs = self.model(spectrum = spectrum,
                                             spectrum_wavelengths = spectrum_wavelengths,
                                             spectrum_mask = spectrum_mask,
                                             targeted_tasks = targeted_tasks,
                                             reduced_outputs = reduced_outputs,
                                             only_output_last = True)
                    else:
                        outputs = self.model(spectrum = spectrum,
                                             spectrum_wavelengths = spectrum_wavelengths,
                                             spectrum_mask = spectrum_mask,
                                             reduced_outputs = reduced_outputs,
                                             only_output_last = True)

                    if masking_prediction:
                        if spectrum_mask is not None:
                            if 'mask' in task_now:
                                target = target[:, spectrum_mask]

                    output_dict = {}
                    for item in outputs:
                        if item == 'tasks_prediction':
                            if outputs[item] is not None:
                                for task in outputs[item]:
                                    task_output = outputs[item][task]
                                    output_dict['tasks_prediction[:=:]{0}'.format(task)] = task_output
                        else:
                            if outputs[item] is not None:
                                task_output = outputs[item]
                                output_dict[item] = task_output

                    all_outputs.append(output_dict)
                    if multi_task_together:
                        for task_idx in range(len(tasks)):
                            tname = tasks[task_idx]
                            all_targets[tname].append(multi_targets[task_idx])
                    else:
                        all_targets.append(target)

                    if rank == 0:
                        eval_progress_bar.update(1)

                del spectrum, spectrum_wavelengths, outputs
                del output_dict
                if multi_task_together:
                    del multi_targets
                else:
                    del target

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                gathered_outputs = self._gather_output_dict(all_outputs, rank)
                if multi_task_together:
                    targets = {}
                    for tname in tasks:
                        targets[tname] = self._gather_target_tensors(all_targets[tname], rank)
                else:
                    targets = self._gather_target_tensors(all_targets, rank)

                model_outputs, tasks_predictions = {}, {}
                if gathered_outputs is not None:
                    for item in gathered_outputs:
                        if 'tasks_prediction' in item:
                            task_name = item.replace('tasks_prediction[:=:]', '')
                            tasks_predictions[task_name] = gathered_outputs[item]
                        else:
                            model_outputs[item] = gathered_outputs[item]

                    if len(tasks_predictions) > 0:
                        model_outputs['tasks_prediction'] = tasks_predictions

                    del tasks_predictions

                    if self.model_wrapped is None:
                        model_outputs = OutputObject[type(self.model)](**model_outputs)
                    else:
                        model_outputs = OutputObject[type(self.model_wrapped)](**model_outputs)

                    if multi_task_together:
                        for tname in tasks:
                            task_type = self.tokenizer.get_task_type(tname)
                            loss_value = self.compute_loss(model_outputs, targets[tname], tname, task_type)
                            metrics, report = self.compute_metrics(model_outputs, targets[tname], tname, task_type)
                            metrics['loss'] = loss_value.item()

                            task_metrics[tname] = metrics
                            task_reports[tname] = report
                    else:
                        if task_now in self.tokenizer.tasks:
                            task_type = self.tokenizer.get_task_type(task_now)
                        else:
                            if display_folder == 'eval':
                                task_type = self.eval_task
                            elif display_folder == 'test':
                                task_type = self.test_task
                            else:
                                task_type = None

                        loss_value = self.compute_loss(model_outputs, targets, task_now, task_type)
                        metrics, report = self.compute_metrics(model_outputs, targets, task_now, task_type)
                        metrics['loss'] = loss_value.item()

                        task_metrics[task_now] = metrics
                        task_reports[task_now] = report

                    del loss_value, metrics, report

                del model_outputs, targets
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if rank == 0:
                    eval_progress_bar.update(1)

                if multi_task_together:
                    break

        aggregated_metircs, aggregated_reports = None, None
        if rank == 0:
            if is_multi_task:
                aggregated_metircs = task_metrics
                aggregated_reports = task_reports
            else:
                for task in task_metrics:
                    aggregated_metircs = task_metrics[task]

                for task in task_reports:
                    aggregated_reports = task_reports[task]

        if display_folder == 'eval':
            self.eval_metrics = aggregated_metircs
            self.eval_reports = aggregated_reports
        else:
            self.test_metrics = aggregated_metircs
            self.test_reports = aggregated_reports

        if rank == 0:
            eval_progress_bar.close()

        if env_init:
            self.logger = logger
            self.clean_ddp(rank)

        return None

    def predict(self, 
            test_dataset = None, 
            test_task_name = None,
            test_task = None, 
            custom_loss_function = None,
            custom_metric_evaluation = None,
            world_size = None):

        if custom_loss_function is not None:
            assert callable(custom_loss_function)
            self.custom_loss_function = custom_loss_function

        if custom_metric_evaluation is not None:
            assert callable(custom_metric_evaluation)
            self.custom_metric_evaluation = custom_metric_evaluation

        default_computing = True
        if (self.custom_loss_function is not None) and (self.custom_metric_evaluation is not None):
            default_computing = False

        if world_size is None:
            world_size = self.world_size
        else:
            if torch.cuda.is_available():
                if self.ddp_backend == 'nccl':
                    if world_size > torch.cuda.device_count():
                        raise ValueError('Cannot set world_size more than' + \
                                ' the number of CUDA devices.')

        if test_dataset is None:
            test_dataset = self.test_dataset
        else:
            assert isinstance(test_dataset, Dataset)

        if test_task_name is None:
            test_task_name = self.test_task_name

        if test_task is None:
            test_task = self.test_task

        if self.dataset_random_sampling:
            if coffee_database_available:
                raise_error = False
                if not isinstance(test_dataset, EnsembleCoffeeDatabaseDataset):
                    raise_error = True
                else:
                    if isinstance(test_dataset, MultiTaskingEnsembleDataset):
                        raise_error = True

                if raise_error:
                    raise NotImplementedError('Please set dataset_random_sampling=False,' + \
                            ' other dataset was not implemented the random_sampling.')
            else:
                raise ValueError('Argument: dataset_random_sampling is designed for dataset in' + \
                       ' private library: coffee_database, set it to False if using the custom dataset.')

        self.setup_dist_env()
        mp.spawn(self._distributed_test,
                 args = (world_size, test_dataset, test_task_name, test_task, 'test', True),
                 nprocs = world_size,
                 join = True)

        self.logger = None

        return None

    def _distributed_test(self,
            rank, 
            world_size, 
            test_dataset, 
            test_task_name, 
            test_task, 
            display_folder = 'test',
            env_init = True):

        return self._distributed_eval(rank = rank,
                                      world_size = world_size,
                                      eval_dataset = test_dataset,
                                      eval_task_name = test_task_name,
                                      eval_task = test_task,
                                      display_folder = display_folder,
                                      env_init = env_init)

    def get_learning_rate(self):
        rate = None
        if self.lr_scheduler is not None:
            rate = self.lr_scheduler.get_last_lr()[0]

        return rate

    def save_model(self, output_dir = None, extra_name = '', rank = None):
        if output_dir is None:
            output_dir = self.output_dir

        assert isinstance(output_dir, str)
        if self.model_wrapped is not None:
            saved_model = self.model_wrapped
        else:
            saved_model = self.model

        assert isinstance(extra_name, str)
        processor_filename = 'processor'
        model_filename = 'model'

        if len(extra_name) > 0:
            processor_filename += '_{0}'.format(extra_name)
            model_filename += '_{0}'.format(extra_name)

        if rank is None:
            saved_model.save_pretrained(output_dir,
                                        processor_filename = processor_filename,
                                        filename = model_filename)
        else:
            if rank == 0:
                saved_model.save_pretrained(output_dir,
                                            processor_filename = processor_filename,
                                            filename = model_filename)

                if self.ddp_backend == 'nccl':
                    saved_model = saved_model.to(rank)

        return None

    def compute_gradient_norm(self):
        if self.model_wrapped is not None:
            targeted_model = self.model_wrapped
        else:
            targeted_model = self.model

        total_norm = None
        for p in targeted_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                if total_norm is None:
                    total_norm = torch.tensor(0.).to(param_norm.device)

                total_norm += param_norm ** 2

        total_norm = total_norm ** 0.5

        return total_norm

    @abc.abstractmethod
    def compute_loss(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_metrics(self):
        raise NotImplementedError()


class CoffeeDatabaseTrainer(BaseTrainer):
    def __init__(self, 
            args,
            dataset = None,
            dataset_task_name = None,
            dataset_task = None,
            model = None):

        if not coffee_database_available:
            raise ImportError('The extended private library: coffee_database does not installed.')

        if dataset is None:
            raise ValueError('Please set the dataset in CoffeeDatabase for training IRBert.')
        else:
            assert isinstance(dataset, (CoffeeDatabaseDataset, EnsembleCoffeeDatabaseDataset))

        available_tasks = ['regression', 'multiclass_classification', 'multilabel_classification']
        if isinstance(dataset, EnsembleCoffeeDatabaseDataset):
            dataset_task = 'auto'
            dataset_task_name = 'auto'
        else:
            if dataset_task is not None:
                assert dataset_task != 'auto'
                assert dataset_task in available_tasks

            assert isinstance(dataset_task_name, str)
            assert len(dataset_task_name) > 0

        if dataset_task is None:
            raise ValueError('Please set argument: dataset_task in the {0}.'\
                    .format(self.__class__.__name__))

        train_dataset = dataset
        eval_dataset = copy.deepcopy(train_dataset)
        test_dataset = copy.deepcopy(train_dataset)
        spectrum_wavelengths = train_dataset.wavelengths

        super(CoffeeDatabaseTrainer, self).__init__(
               args = args,
               spectrum_wavelengths = spectrum_wavelengths,
               train_dataset = train_dataset,
               train_task_name = dataset_task_name,
               train_task = dataset_task, 
               eval_dataset = eval_dataset,
               eval_task_name = dataset_task_name,
               eval_task = dataset_task,
               test_dataset = test_dataset,
               test_task_name = dataset_task_name,
               test_task = dataset_task,
               model = model)

    def compute_loss(self, model_output, targets, task_name, task_type):
        return default_loss_computing(model_output, targets, task_name, task_type)

    def compute_metrics(self, model_output, targets, task_name, task_type):
        return default_metrics_computing(model_output, targets, task_name, task_type)


class TensorDatasetTrainer(BaseTrainer):
    def __init__(self,
            args,
            spectrum_wavelengths = None,
            custom_loss_function = None,
            custom_metric_evaluation = None,
            train_dataset = None,
            train_task_name = None,
            train_task = None,
            eval_dataset = None,
            eval_task_name = None,
            eval_task = None,
            test_dataset = None,
            test_task_name = None,
            test_task = None,
            model = None):

        super(TensorDatasetTrainer, self).__init__(
                args = args,
                spectrum_wavelengths = spectrum_wavelengths,
                custom_loss_function = custom_loss_function,
                custom_metric_evaluation = custom_metric_evaluation,
                train_dataset = train_dataset,
                train_task_name = train_task_name,
                train_task = train_task,
                eval_dataset = eval_dataset,
                eval_task_name = eval_task_name,
                eval_task = eval_task,
                test_dataset = test_dataset,
                test_task_name = test_task_name,
                test_task = test_task,
                model = model)

    def compute_loss(self, model_output, targets, task_name, task_type):
        if self.custom_loss_function is None:
            loss_value = default_loss_computing(model_output, targets, task_name, task_type)
        else:
            loss_value = self.custom_loss_function(model_output, targets)

        return loss_value

    def compute_metrics(self, model_output, targets, task_name, task_type):
        report = None
        if self.custom_metric_evaluation is None:
            (metrics,
             report) = default_metrics_computing(model_output, targets, task_name, task_type)
        else:
            metrics = self.custom_metric_evaluation(model_output, targets)
            assert isinstance(metrics, dict)
            for m in metrics:
                if not isinstance(m, str):
                    raise ValueError('The key of the metric dict must be a Python string object.')

                assert len(m) > 0

        return metrics, report


