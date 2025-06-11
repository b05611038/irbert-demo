import torch


__all__ = ['table_like_layout_line', 'r2_score']


def table_like_layout_line(content, items, block_length, percentage = False):
    assert len(content) == len(items)

    item_line, content_line = '', ''
    for item in items:
        item_line += '{0}'.format(item).rjust(block_length)

    for value in content:
        if percentage:
            value *= 100.
            content_line += '{0:.2f}%'.format(value).rjust(block_length)
        else:
            content_line += '{0:.4f}'.format(value).rjust(block_length)

    return item_line, content_line

def r2_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape

    mean_y_true = torch.mean(y_true, dim = 0)

    ss_tot = torch.sum((y_true - mean_y_true) ** 2, dim = 0)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim = 0)

    r2 = 1 - (ss_res / ss_tot)

    return r2
    

