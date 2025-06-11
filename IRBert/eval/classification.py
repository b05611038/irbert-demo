import copy

import torch
import torch.nn as nn

from .base import BaseReport
from .utils import table_like_layout_line

from ..utils import save_as_csv


__all__ = ['MultiClassReport', 'MultiLabelReport']


class MultiClassReport(BaseReport):
    def __init__(self, prediction, ground_truth, 
            metrics = ('accuracy', 'acc_per_class', 'cross_entropy'),
            class_names = None,
            percentage = True):

        assert isinstance(prediction, torch.Tensor)
        assert isinstance(ground_truth, torch.Tensor)

        assert prediction.shape[0] == ground_truth.shape[0]

        self.class_names = class_names
        self.percentage = percentage

        super(MultiClassReport, self).__init__(
                prediction = prediction,
                ground_truth = ground_truth,
                metrics = metrics)

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, class_names):
        if class_names is not None:
            assert isinstance(class_names, (tuple, list))

        self._class_names = class_names
        return None

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, percentage):
        assert isinstance(percentage, bool)
        self._percentage = percentage
        return None
 
    def evaluate(self, prediction, ground_truth):
        device = prediction.device
        prediction = prediction.detach().clone()
        ground_truth = ground_truth.detach().clone()

        self.data_number = prediction.shape[0]

        results = {}
        cross_entropy_eval = nn.CrossEntropyLoss()
        cross_entropy = cross_entropy_eval(prediction, ground_truth)
        results['cross_entropy'] = cross_entropy.detach().cpu().item()

        _, predicted_classes = torch.max(prediction, dim = 1)
        correct_predictions = (predicted_classes == ground_truth.view(-1))
        accuracy = correct_predictions.sum() / correct_predictions.shape[0]
        results['accuracy'] = accuracy.cpu().item()

        acc_per_class  = torch.zeros(prediction.shape[1], device = device)
        for class_idx in range(prediction.shape[1]):
            class_mask = ground_truth.view(-1) == class_idx
            class_acc = (predicted_classes[class_mask] == ground_truth.view(-1)[class_mask]).sum()
            class_total = class_mask.sum()

            acc_per_class[class_idx] = class_acc / (class_total + self.eps)

        results['acc_per_class'] = acc_per_class.cpu().tolist()

        return results

    def summary(self, layout_func = print):
        lines = ['=' * self.terminal_length, 'Multi-class classification report: ']
        lines.append('Data number: {0}'.format(self.data_number))
        lines.append('Class Acc.: ')

        item_number_per_line = self.terminal_length // self.block_length
        class_idx = 0
        while class_idx < len(self.results['acc_per_class']):
            end_idx = class_idx + item_number_per_line
            if end_idx > len(self.results['acc_per_class']):
                end_idx = len(self.results['acc_per_class'])

            content = self.results['acc_per_class'][class_idx: end_idx]
            if self.class_names is None:
                categories = ['Class {0}'.format(i + 1) for i in range(class_idx, end_idx)]
            else:
                categories = self.class_names[class_idx: end_idx]

            (category_line, 
             content_line) = table_like_layout_line(content, 
                                                    categories, 
                                                    self.block_length,
                                                    self.percentage)

            lines.append(category_line)
            lines.append(content_line)
            class_idx = end_idx

        lines.append('')
        if self.percentage:
             display_acc = self.results['accuracy'] * 100.
             lines.append('Total Acc.: {0:.2f}% | Cross entropy: {1:.4f}'\
                     .format(display_acc, self.results['cross_entropy']))
        else:
             lines.append('Total Acc.: {0:.4f} | Cross entropy: {1:.4f}'\
                     .format(self.results['accuracy'], self.results['cross_entropy']))
 
        lines.append('=' * self.terminal_length)
        for l in lines:
            layout_func(l)

        return None

    def save_as_csv(self, filename):
        if self.class_names is None:
            col_names = ['Class {0} acc.'.format(i + 1) for i in range(len(self.results['acc_per_class']))]
        else:
            col_names = []
            for name in self.class_names:
                col_names.append(name + ' acc.')

        table = [self.results['cross_entropy'], self.results['accuracy']] + self.results['acc_per_class']
        table = torch.tensor(table).unsqueeze(0)
        column_names = ['cross_entropy', 'total_accuracy']
        column_names += col_names

        save_as_csv(table, filename, column_names = column_names)

        return None


class MultiLabelReport(BaseReport):
    def __init__(self, prediction, ground_truth,
            metrics = ('accuracy', 'acc_per_class', 'recall', 'recall_per_class', 
            'precision', 'precision_per_class', 'f1', 'f1_per_class', 'binary_cross_entropy'),
            prediction_boundary = 0.5,
            class_names = None,
            percentage = True):

        assert isinstance(prediction, torch.Tensor)
        assert isinstance(ground_truth, torch.Tensor)

        assert prediction.shape[0] == ground_truth.shape[0]
        assert prediction.shape[1] == ground_truth.shape[1]

        assert prediction.max() <= 1. and prediction.min() >= 0.
        assert ground_truth.max() <= 1. and ground_truth.min() >= 0.

        self.prediction_boundary = prediction_boundary
        self.class_names = class_names
        self.percentage = percentage

        super(MultiLabelReport, self).__init__(
                prediction = prediction,
                ground_truth = ground_truth,
                metrics = metrics)

    @property
    def prediction_boundary(self):
        return self._prediction_boundary

    @prediction_boundary.setter
    def prediction_boundary(self, prediction_boundary):
        assert isinstance(prediction_boundary, float)
        assert prediction_boundary <= 1. and prediction_boundary >= 0.
        self._prediction_boundary = prediction_boundary
        return None

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, class_names):
        if class_names is not None:
            assert isinstance(class_names, (tuple, list))

        self._class_names = class_names
        return None

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, percentage):
        assert isinstance(percentage, bool)
        self._percentage = percentage
        return None

    def evaluate(self, prediction, ground_truth):
        device = prediction.device
        prediction = prediction.detach().clone()
        ground_truth = ground_truth.detach().clone()

        self.data_number = prediction.shape[0]

        results = {}
        cross_entropy_eval = nn.BCELoss()
        if isinstance(prediction, (torch.LongTensor, torch.cuda.LongTensor)):
            prediction = prediction.float()

        if isinstance(ground_truth, (torch.LongTensor, torch.cuda.LongTensor)):
            ground_truth = ground_truth.float()

        binary_cross_entropy = cross_entropy_eval(prediction, ground_truth)

        results['binary_cross_entropy'] = binary_cross_entropy.detach().cpu().item()

        prediction = (prediction > self.prediction_boundary).float()

        class_recall_total = torch.sum(ground_truth, dim = 0) # TP + FN
        class_recall_error = ground_truth - prediction
        class_recall_error[class_recall_error < 0.] = 0.
        class_recall_error = torch.sum(class_recall_error, dim = 0) # FN
        class_recall = 1. - (class_recall_error / (class_recall_total + self.eps))
        recall_total = torch.sum(class_recall_total) 
        recall_error = torch.sum(class_recall_error)
        recall = 1. - (recall_error / (recall_total + self.eps))

        class_fn = class_recall_error.clone()
        class_tp = class_recall_total - class_fn

        class_precision_total = torch.sum(prediction, dim = 0) # TP + FP
        class_precision_error = prediction - ground_truth
        class_precision_error[class_precision_error < 0.] = 0. 
        class_precision_error =  torch.sum(class_precision_error, dim = 0) # FP
        class_precision = 1. - (class_precision_error / (class_precision_total + self.eps))
        precision_total = torch.sum(class_precision_total)
        precision_error = torch.sum(class_precision_error)
        precision = 1. - (precision_error / (precision_total + self.eps))

        class_fp = class_precision_error.clone()
        class_tn = ground_truth.shape[0] - (class_fn + class_tp + class_fp)

        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        
        class_accuracy_total = torch.sum(torch.ones_like(ground_truth), dim = 0)
        class_accuracy_error = torch.sum(torch.abs(ground_truth - prediction), dim = 0)
        class_accuracy = 1. - (class_accuracy_error / (class_accuracy_total + self.eps))
        accuracy_total = torch.sum(class_accuracy_total)
        accuracy_error = torch.sum(class_accuracy_error)
        accuracy = 1. - (accuracy_error / (accuracy_total + self.eps))

        class_accuracy = class_accuracy.cpu().tolist()
        class_recall = class_recall.cpu().tolist()
        class_precision = class_precision.cpu().tolist()
        class_f1 = class_f1.cpu().tolist()

        results['accuracy'] = accuracy.cpu().item()
        results['acc_per_class'] = class_accuracy
        results['recall'] = recall.cpu().item()
        results['recall_per_class'] = class_recall
        results['precision'] = precision.cpu().item()
        results['precision_per_class'] = class_precision
        results['f1'] = f1.cpu().item()
        results['f1_per_class'] = class_f1

        return results

    def _class_summary(self, lines, metric):
        class_idx = 0
        item_number_per_line = self.terminal_length // self.block_length
        while class_idx < len(self.results[metric]):
            end_idx = class_idx + item_number_per_line
            if end_idx > len(self.results[metric]):
                end_idx = len(self.results[metric])

            content = self.results[metric][class_idx: end_idx]
            if self.class_names is None:  
                categories = ['Class {0}'.format(i + 1) for i in range(class_idx, end_idx)]
            else:
                categories = self.class_names[class_idx: end_idx]

            if metric == 'f1_per_class':
                percentage = False
            else:
                percentage = self.percentage

            (category_line,
             content_line) = table_like_layout_line(content,
                                                    categories,
                                                    self.block_length,
                                                    percentage)

            lines.append(category_line)
            lines.append(content_line)
            class_idx = end_idx

        return lines

    def summary(self, layout_func = print):
        lines = ['=' * self.terminal_length, 'Multi-label classification report: ']
        lines.append('Data number: {0}'.format(self.data_number))

        item_number_per_line = self.terminal_length // self.block_length

        lines.append('')
        lines.append('Class Acc.: ')
        lines = self._class_summary(lines, 'acc_per_class')
        lines.append('')
        lines.append('Class Recall: ')
        lines = self._class_summary(lines, 'recall_per_class')
        lines.append('')
        lines.append('Class Precision: ')
        lines = self._class_summary(lines, 'precision_per_class')
        lines.append('')
        lines.append('Class F1: ')
        lines = self._class_summary(lines, 'f1_per_class')

        lines.append('')
        lines.append('Binary cross entropy: {0:.4f}'.format(self.results['binary_cross_entropy']))
        if self.percentage:
             display_acc = self.results['accuracy'] * 100.
             display_recall = self.results['recall'] * 100.
             display_precision = self.results['precision'] * 100.
             lines.append('Accuracy: {0:.2f}% | Recall: {1:.2f}% | Precision: {2:.2f}% | F1: {3:.4f}'\
                     .format(display_acc, display_recall, display_precision, self.results['f1']))
        else:
             lines.append('Accuracy: {0:.4f} | Recall: {1:.4f} | Precision: {2:.4f} | F1: {3:.4f}'\
                     .format(self.results['accuracy'], self.results['recall'],self.results['precision'], 
                     self.results['f1']))
 
        lines.append('=' * self.terminal_length)
        for l in lines:
            layout_func(l)

        return None

    def save_as_csv(self, filename):
        if self.class_names is None:
            col_names = ['Class {0}'.format(i + 1) for i in range(len(self.results['acc_per_class']))]
        else:
            col_names = copy.deepcopy(self.class_names)

        column_names = col_names + ['total', 'loss']
        row_names = ['accuracy', 'recall', 'precision', 'f1']
        table = []
        table.append(self.results['acc_per_class'] + [self.results['accuracy'], 
                self.results['binary_cross_entropy']])
        table.append(self.results['recall_per_class'] + [self.results['recall'], 
                self.results['binary_cross_entropy']])
        table.append(self.results['precision_per_class'] + [self.results['precision'], 
                self.results['binary_cross_entropy']])
        table.append(self.results['f1_per_class'] + [self.results['f1'], 
                self.results['binary_cross_entropy']])
        table = torch.tensor(table)

        save_as_csv(table, filename, row_names = row_names, column_names = column_names)

        return None


