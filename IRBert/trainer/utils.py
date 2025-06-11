import torch
import torch.nn as nn
import torch.nn.functional as F

from ..eval import (MultiClassReport,
                    MultiLabelReport,
                    RegressionReport)

__all__ = ['default_loss_computing', 'default_metric_computing']

def default_loss_computing(model_output, targets, task_name, task_type):
    assert task_type != 'auto'
    if task_type == 'regression':
        loss_func = nn.SmoothL1Loss(beta = 1.)
        if 'reconstruction' in task_name:
            if 'mask' in task_name:
                targeted_model_output = model_output.reconstructed_masked_spectrum
                if targeted_model_output is None:
                    targeted_model_output = model_output.reconstructed_spectrum
            else:
                targeted_model_output = model_output.reconstructed_spectrum

            if targeted_model_output is None:
                targeted_model_output = model_output.reconstructed_spectrum
        else:
            targeted_model_output = model_output.tasks_prediction[task_name]

    elif 'classification' in task_type:
        loss_func = nn.CrossEntropyLoss(label_smoothing = 0.1)
        targeted_model_output = model_output.tasks_prediction[task_name]
        if 'multilabel' in task_type:
            targeted_model_output = targeted_model_output.reshape(-1, 2)
            targets = targets.reshape(-1)

        targets = targets.long().squeeze(-1)
    else:
        raise ValueError('Invalid task_type ({0}) inputted'.format(task_type))

    loss_value = loss_func(targeted_model_output, targets)

    return loss_value

def default_metrics_computing(model_output, targets, task_name, task_type):
    if task_type is None:
        raise RuntimeError('Please use the IRBertForMultiTaskPredtion model in trainer.')

    assert task_type != 'auto'
    metrics = {}
    if task_type == 'regression':
        if 'reconstruction' in task_name:
            if 'mask' in task_name:
                targeted_model_output = model_output.reconstructed_masked_spectrum
                if targeted_model_output is None:
                    targeted_model_output = model_output.reconstructed_spectrum
            else:
                targeted_model_output = model_output.reconstructed_spectrum
        else:
            targeted_model_output = model_output.tasks_prediction[task_name]

        if not torch.isnan(targeted_model_output).any():
            report = RegressionReport(targeted_model_output, targets)
            metrics['MSE'] = sum(report.results['MSE'])
            metrics['MAE'] = sum(report.results['MAE'])
            metrics['RPD'] = sum(report.results['RPD'])

    elif 'classification' in task_type:
        targets = targets.squeeze(-1)
        targeted_model_output = model_output.tasks_prediction[task_name]
        if not torch.isnan(targeted_model_output).any():
            num_classes = targeted_model_output.shape[-1]
            predicted_classes = torch.argmax(targeted_model_output, dim = -1)
            if 'multilabel' in task_type:
                report = MultiLabelReport(predicted_classes.float(), targets)
                metrics['accuracy'] = report.results['accuracy']
                metrics['recall'] = report.results['recall']
                metrics['precision'] = report.results['precision']
                metrics['F1_score'] = report.results['f1']
            else:
                predicted_classes = F.one_hot(predicted_classes, num_classes = num_classes)
                report = MultiClassReport(predicted_classes.float(), targets.long())
                metrics['accuracy'] = report.results['accuracy']

    else:
        raise ValueError('Invalid task_type ({0}) inputted'.format(task_type))

    return metrics, report


