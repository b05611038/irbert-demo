import math
import torch

DEFAULT_FWHM = math.sqrt(3. ** 2 - 2. ** 2)

def resample_spectrum(spectrum, wavelengths, target_wavelengths, fwhm = DEFAULT_FWHM):
    """
    Resample an IR spectrum to a new wavelength resolution using a Gaussian response function in parallel.
    
    Args:
        spectrum (torch.Tensor): The original spectrum (shape: [batch_size, n_original]).
        wavelengths (torch.Tensor): The wavelengths corresponding to the original spectrum (1D tensor).
        target_wavelengths (torch.Tensor): The target wavelengths for resampling (1D tensor).
        fwhm (float): Full width at half maximum for the Gaussian instrument response.
        
    Returns:
        torch.Tensor: Resampled spectra (shape: [batch_size, n_target]).
    """
    # Convert FWHM to standard deviation
    std = fwhm / (2. * math.sqrt(2. * math.log(2.)))
    
    # Compute the pairwise distance between wavelengths and target wavelengths
    # Shape: [n_original, n_target]
    wavelength_differences = wavelengths.unsqueeze(1) - target_wavelengths.unsqueeze(0)
    
    # Compute the Gaussian kernel for all target wavelengths
    # Shape: [n_original, n_target]
    gaussian_kernels = torch.exp(-0.5 * (wavelength_differences / std) ** 2)
    gaussian_kernels = gaussian_kernels / gaussian_kernels.sum(dim=0, keepdim=True)  # Normalize kernels
    
    # Resample spectra using matrix multiplication
    # Shape of spectrum: [batch_size, n_original]
    # Shape of gaussian_kernels: [n_original, n_target]
    # Result: [batch_size, n_target]
    resampled_spectra = torch.matmul(spectrum, gaussian_kernels)
    
    return resampled_spectra


