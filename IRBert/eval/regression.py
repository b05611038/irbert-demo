import torch

from .base import BaseReport
from .utils import (table_like_layout_line,
                    r2_score)

from ..utils import save_as_csv


__all__ = ['RegressionReport']


class RegressionReport(BaseReport):
    def __init__(self, prediction, ground_truth, 
            metrics = ('MSE', 'MAE', 'R2', 'RMSE', 'SE', 'STD', 'RPD'),
            show_detail_eval = False,
            class_names = None):

        assert isinstance(prediction, torch.Tensor)
        assert isinstance(ground_truth, torch.Tensor)

        assert prediction.shape[0] == ground_truth.shape[0]
        assert prediction.shape[1] == ground_truth.shape[1]

        self.show_detail_eval = show_detail_eval
        self.class_names = class_names

        super(RegressionReport, self).__init__(
                prediction = prediction,
                ground_truth = ground_truth,
                metrics = metrics)

    @property
    def show_detail_eval(self):
        return self._show_detail_eval

    def show_detail_eval(self, show_detail_eval):
        assert isinstance(show_detail_eval, bool)
        self._show_detail_eval = show_detail_eval
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

    def evaluate(self, prediction, ground_truth):
        prediction = prediction.detach().clone()
        ground_truth = ground_truth.detach().clone()

        self.data_number = prediction.shape[0]

        results = {}

        standard_deviation = torch.std(prediction, dim = 0, unbiased = True)

        error = prediction - ground_truth
        mean_square_error = (error ** 2).mean(dim = 0)
        root_mean_square_error = mean_square_error ** 0.5
        mean_average_error = torch.abs(error).mean(axis = 0)
        standard_error = torch.std(error, dim = 0, unbiased = True)
        rpd = standard_deviation / standard_error

        r_square = r2_score(prediction, ground_truth)
        results['MSE'] = mean_square_error.cpu().tolist()
        results['MAE'] = mean_average_error.cpu().tolist()
        results['R2'] = r_square.cpu().tolist()
        results['RMSE'] = root_mean_square_error.cpu().tolist()
        results['SE'] = standard_error.cpu().tolist()
        results['RPD'] = rpd.cpu().tolist()

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

            (category_line,
             content_line) = table_like_layout_line(content,
                                                    categories,
                                                    self.block_length,
                                                    False)

            lines.append(category_line)
            lines.append(content_line)
            class_idx = end_idx

        return lines

    def summary(self, layout_func = print):
        lines = ['=' * self.terminal_length, 'Regression report: ']
        lines.append('Data number: {0}'.format(self.data_number))

        item_number_per_line = self.terminal_length // self.block_length

        if self.show_detail_eval:
            lines.append('')
            lines.append('Class RMSE: ')
            lines = self._class_summary(lines, 'RMSE')
            lines.append('')
            lines.append('Class SE: ')
            lines = self._class_summary(lines, 'SE')
            lines.append('')
            lines.append('Class R2: ')
            lines = self._class_summary(lines, 'R2')
            lines.append('')
            lines.append('Class RPD: ')
            lines = self._class_summary(lines, 'RPD')
            lines.append('')

        lines.append('Class MSE: ')
        lines = self._class_summary(lines, 'MSE')
        lines.append('')
        lines.append('Class MAE: ')
        lines = self._class_summary(lines, 'MAE')
        lines.append('')

        lines.append('=' * self.terminal_length)
        for l in lines:
            layout_func(l) 

        return None

    def save_as_csv(self, filename):
        if self.class_names is None:
            row_names = ['Class {0}'.format(i + 1) for i in range(len(self.results['MSE']))]
        else:
            row_names = copy.deepcopy(self.class_names)

        column_names = ['MSE', 'MAE', 'RPD']
        if self.show_detail_eval:
            column_names += ['RMSE', 'SE', 'R2']

        table = []
        table.append(self.results['MSE'])
        table.append(self.results['MAE'])
        table.append(self.results['RPD'])
        if self.show_detail_eval:
             table.append(self.results['RMSE'])
             table.append(self.results['SE'])
             table.append(self.results['R2'])

        table = torch.tensor(table).permute(1, 0)
        
        save_as_csv(table, filename, row_names = row_names, column_names = column_names)

        return None


