import torch
import math
import numpy as np

@torch.jit.script
def gaussian(window_size: int, tfdiff: float):
    gaussian = torch.tensor([math.exp(-(x - window_size//2)**2/float(2*tfdiff**2)) for x in range(window_size)])
    return gaussian / gaussian.sum()


@torch.jit.script
def create_window(height: int, width: int):
    h_window = gaussian(height, 1.5).unsqueeze(1)
    w_window = gaussian(width, 1.5).unsqueeze(1)
    _2D_window = h_window.mm(w_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(1, 1, height, width).contiguous()
    return window


def eval_ssim(pred, data, height, width, device):
    window = create_window(height, width).to(torch.complex64).to(device)
    padding = [height//2, width//2]
    mu_pred = torch.nn.functional.conv2d(pred, window, padding=padding, groups=1)
    mu_data = torch.nn.functional.conv2d(data, window, padding=padding, groups=1)
    mu_pred_pow = mu_pred.pow(2.)
    mu_data_pow = mu_data.pow(2.)
    mu_pred_data = mu_pred * mu_data
    tfdiff_pred = torch.nn.functional.conv2d(pred*pred, window, padding=padding, groups=1) - mu_pred_pow
    tfdiff_data = torch.nn.functional.conv2d(data*data, window, padding=padding, groups=1) - mu_data_pow
    tfdiff_pred_data = torch.nn.functional.conv2d(pred*data, window, padding=padding, groups=1) - mu_pred_data
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu_pred*mu_data+C1) * (2*tfdiff_pred_data.real+C2)) / ((mu_pred_pow+mu_data_pow+C1)*(tfdiff_pred+tfdiff_data+C2))
    return 2*ssim_map.mean().real


def real_ssim(array1, array2):
    # Calculate mean, variance, and covariance
    # Constants for SSIM calculation
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    ux = array1.mean()
    uy = array2.mean()
    var_x = array1.var()
    var_y = array2.var()
    cov_xy = np.cov(array1.flatten(), array2.flatten())[0, 1]

    # Calculate SSIM components
    A1 = 2 * ux * uy + C1
    A2 = 2 * cov_xy + C2
    B1 = ux ** 2 + uy ** 2 + C1
    B2 = var_x + var_y + C2
    
    # print(A1, A2, B1, B2)

    # Calculate SSIM index
    ssim_index = (A1 * A2) / (B1 * B2)
    return ssim_index