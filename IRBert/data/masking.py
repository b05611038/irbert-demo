import math
import torch


__all__ = ['IRSpectrumMaskGenerator']


class IRSpectrumMaskGenerator:
    def __init__(self, masked_ratio):
        assert isinstance(masked_ratio, float)
        assert masked_ratio >= 0. and masked_ratio <= 1.
        self.masked_ratio = masked_ratio

    def __repr__(self):
        return '{0}(masked_ratio={1})'.format(self.__class__.__name__,
                self.masked_ratio)

    def __call__(self, spectrum):
        wave_numebr = spectrum.shape[-1]
        mask = self.generate(wave_numebr)
        mask = mask.to(spectrum.device)
        return mask

    def generate(self, wave_number):
        assert isinstance(wave_number, int)
        mask_number = math.ceil(self.masked_ratio * float(wave_number))

        random_sampled = torch.randperm(wave_number, dtype = torch.long)[: mask_number]
        mask, _ = torch.sort(random_sampled)
        return mask


