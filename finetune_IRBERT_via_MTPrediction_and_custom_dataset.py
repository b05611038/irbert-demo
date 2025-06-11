"""
Demo: Finetune IR-BERT on a custom dataset with synthetic spectrum data.

This script demonstrates how to use IR-BERT for multi-task prediction,
including custom loss, evaluation metrics, and linear probing.

⚠️ Processor configuration and core embedding logic are redacted in this demo version.
"""

import os
import random

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from IRBert import (IRBertConfig,
                    IRBertForMultiTaskPrediction,
                    TrainingArgument,
                    TensorDatasetTrainer)


def my_loss_function(irbert_output, target):
    # Following can be defined by yourself, all task predictions are in the output object.
    task_name = list(irbert_output.tasks_prediction.keys())[0]
    targeted_output = irbert_output.tasks_prediction[task_name]
    if task_name == 'my_regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        if task_name == 'my_multi-label_classification':
            targeted_output = targeted_output.reshape(-1, 2)
            target = target.long().view(-1)

    return criterion(targeted_output, target)

def my_metrics(irbert_output, target):
    # Following can be defined by yourself, all task predictions are in the output object.
    # must return a Python dict object.
    task_name = list(irbert_output.tasks_prediction.keys())[0]
    targeted_output = irbert_output.tasks_prediction[task_name]
    if task_name == 'my_regression':
        error = targeted_output - target
        mae = torch.abs(error).mean().cpu().item()
        mse = (error ** 2).mean().cpu().item()
        metrics = {'MAE': mae, 'MSE': mse}
    else:
        if task_name == 'my_multi-label_classification':
            targeted_output = targeted_output.reshape(-1, 2)
            target = target.long().view(-1)

        _, predicted_classes = torch.max(targeted_output, dim = 1)
        correct_predictions = (predicted_classes == target.view(-1))
        accuracy = correct_predictions.sum() / correct_predictions.shape[0]
        metrics = {'accuracy': accuracy}

    return metrics

def main():
    train_steps = 10000
    learning_rate = 5e-6
    weight_decay = 0.001
    pretrained_model = 'my-irbert'
    finetuned_model_name = 'my-finetuned-irbert'
    linear_probing = False

    # please load your data to construct your custom dataset
    train_sample_number = 1000
    eval_sample_number = 100
    test_sample_number = 200
    wavelengths = [i for i in range(700, 2500, 2)]
    train_spectra = torch.randn(train_sample_number, len(wavelengths))
    eval_spectra = torch.randn(eval_sample_number, len(wavelengths))
    test_spectra = torch.randn(test_sample_number, len(wavelengths))
    wavelengths = torch.tensor(wavelengths, dtype = torch.float)

    # your target, here gives examples of regresion, multi-class classification
    # Following is an example of regression task
    column_numbers = 5
    train_label = torch.randn(train_sample_number, column_numbers)
    eval_label = torch.randn(eval_sample_number, column_numbers)
    test_label = torch.randn(test_sample_number, column_numbers)
    task_name = 'my_regression'
    task_content = {'task_type': 'regression',
                    'target_classes': ['item{0}'.format(i) for i in range(column_numbers)]}

    # Following is an example of multi-class classificaiton task
    # class_nums = 10
    # train_label = [random.randrange(0, class_nums) for _ in range(train_sample_number)]
    # train_label = torch.tensor(train_label, dtype = torch.long)
    # eval_label = [random.randrange(0, class_nums) for _ in range(eval_sample_number)]
    # eval_label = torch.tensor(eval_label, dtype = torch.long)
    # test_label = [random.randrange(0, class_nums) for _ in range(test_sample_number)]
    # test_label = torch.tensor(test_label, dtype = torch.long)
    # task_name = 'my_multi-class_classification'
    # task_content = {'task_type': 'multiclass_classification',
    #                 'target_classes': ['class{0}'.format(i) for i in range(class_nums)]}

    # Following is an example of multi-label classificaiton task
    # class_nums = 10
    # train_label = [[random.randint(0, 1) for j in range(class_nums)] for i in range(train_sample_number)]
    # train_label = torch.tensor(train_label, dtype = torch.float)
    # eval_label = [[random.randint(0, 1) for j in range(class_nums)] for i in range(eval_sample_number)]
    # eval_label = torch.tensor(eval_label, dtype = torch.float)
    # test_label = [[random.randint(0, 1) for j in range(class_nums)] for i in range(test_sample_number)]
    # test_label = torch.tensor(test_label, dtype = torch.float)
    # task_name = 'my_multi-label_classification'
    # task_content = {'task_type': 'multilabel_classification',
    #                 'target_classes': ['class{0}'.format(i) for i in range(class_nums)]}

    # Initial the TensorDatasets
    train_dataset = TensorDataset(train_spectra, train_label)
    eval_dataset = TensorDataset(eval_spectra, eval_label)
    test_dataset = TensorDataset(test_spectra, test_label)

    # define model config (this is the base version.)
    # only work when train from scratch 
    model_config = IRBertConfig(wavelength_window_size = 25.,
                                num_hidden_layer = 12,
                                num_hidden_head = 12,
                                hidden_size = 768,
                                feedforward_size = 3024,
                                conv_dim = (192, 384, 768))

    model = IRBertForMultiTaskPrediction(model_config)
    # if load pre-trained model, please use the following line
    model = model.from_pretrained(pretrained_model)

    if linear_probing:
        for param in model.processor.parameters():
            param.requires_grad = False

        for param in model.model.parameters():
            param.requires_grad = False

    model.add_task(task_name, **task_content)

    arguments = TrainingArgument(output_dir = finetuned_model_name,
                                 train_steps = train_steps,
                                 eval_steps = -1, # no evaluation when it set to -1
                                 test_steps = -1, # no testing when it set to -1
                                 save_steps = -1, # not save model during training when it set to -1
                                 learning_rate = learning_rate,
                                 weight_decay = weight_decay)

    # The [set]_task, [set]_task_name are argument used in default setting
    # If custom_loss_function and custom_metric_evaluation are set, 
    # actually, you don't need to set these arguments
    trainer = TensorDatasetTrainer(args = arguments,
                                   model = model,
                                   spectrum_wavelengths = wavelengths,
                                   train_dataset = train_dataset,
                                   eval_dataset = eval_dataset,
                                   test_dataset = test_dataset,
                                   custom_loss_function = my_loss_function,
                                   custom_metric_evaluation = my_metrics)

    trainer.train()
    model = trainer.model
    model.save_pretrained(finetuned_model_name)

    print('Finetune model: {0} done.'.format(finetuned_model_name))

    return None

    
if __name__ == '__main__':
    main()


