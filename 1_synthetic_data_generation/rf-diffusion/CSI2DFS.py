import numpy as np
import struct
from scipy.signal import butter, lfilter, get_window
from scipy.fft import fft, fftshift
import pywt
import os
import torch
import torch.nn.functional as F


def complex_pca(data, n_components=2):
    # Center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Compute the covariance matrix
    cov_matrix = np.dot(centered_data.T.conj(), centered_data) / (len(data) - 1)
    
    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top n_components
    principal_components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    
    return principal_components, explained_variance, mean


def gaussian_window(N, K=0.005):
    # Compute the standard deviation for the desired tail value K
    sigma = np.sqrt(-N**2 / (8 * np.log(K)))
    
    # Using scipy to generate Gaussian window
    # scipy uses std deviation directly
    gauss_window = get_window(('gaussian', sigma), N)
    return gauss_window

def get_euclidean_norm(x):
    return np.sqrt(np.sum(np.square(np.abs(x))))


def tfrsp(x, t_indices, N, h):
    x_len = len(x)
    # prepare the output matrix
    tfr = np.zeros((N, x_len), dtype=np.complex_)
    Lh = (len(h)-1) // 2
    # conjugate the window
    h = np.conj(h)
    
    # loop over the time indices
    for idx, ti in enumerate(t_indices):
        # compute delta relative to current time index
        tau = np.arange(-min(N//2-1, Lh, ti), min(N//2, Lh+1, x_len-ti))
        output_indices = (N + tau) % N
        x_indices = ti + tau
        window_indices = Lh + tau
        # apply the window
        windowed_data = np.zeros(N, dtype=np.complex_)
        windowed_data[output_indices] = x[x_indices] * \
            h[window_indices] / get_euclidean_norm(h[window_indices])
        
        # FFT of the windowed data
        tfr[:, idx] = fft(windowed_data, n=N)
        # if idx == 125:
        #     print(output_indices)
        #     print(windowed_data[0:30])
        #     print(tfr[:, idx])
    
    # compute the power spectrum
    tfr = np.abs(tfr) ** 2
    # frequency vector
    if N % 2 == 0:
        f = np.arange(0,N/2+1).tolist() + np.arange(-N/2,0).tolist()
    else:
        f = np.arange(0, (N+1)/2).tolist() + np.arange(-(N-1)/2,0).tolist()
    
    return tfr, t_indices, np.array(f) / N


# the input CSI has been reshaped as [512,90] and normalized with zero mean and standard variance
# we need ori_packet_cnt to calculate the samp_rate
# the window_size is 512//4 + 1
def get_doppler_spectrum(csi_data, ori_packet_cnt, method, ant_cnt=3, subcarrier_cnt=30):
    new_len = 512
    window_size = new_len // 4 + 1
    ori_samp_rate = 1000
    new_samp_rate = int(new_len / (ori_packet_cnt / ori_samp_rate))
    new_samp_rate = (new_samp_rate // 2) * 2
    
    half_rate = new_samp_rate / 2
    uppe_orde = 6
    uppe_stop = 60
    lowe_orde = 3
    lowe_stop = 2
    lu, ld = butter(uppe_orde, uppe_stop / half_rate, 'low')
    hu, hd = butter(lowe_orde, lowe_stop / half_rate, 'high')
    freq_bins_unwrap = np.concatenate( [np.arange(0,new_samp_rate/2), np.arange(-new_samp_rate/2,0)], axis=0) / new_samp_rate
    freq_lpf_sele = (freq_bins_unwrap <= uppe_stop / new_samp_rate) & (freq_bins_unwrap >= -uppe_stop / new_samp_rate)
    freq_lpf_positive_max = np.sum(freq_lpf_sele[:len(freq_lpf_sele)//2])
    freq_lpf_negative_min = np.sum(freq_lpf_sele[len(freq_lpf_sele)//2:])
    
    # select antenna pair [WiDance]
    csi_mean = np.mean(np.abs(csi_data), axis=0)
    csi_var = np.sqrt(np.var(np.abs(csi_data), axis=0))
    csi_mean_var_ratio = csi_mean / csi_var
    # print("csi_mean", csi_mean)
    # print("csi_var", csi_var)
    # print("csi_mean_var_ratio", csi_mean_var_ratio)
    
    # [NOTE] the different storage order between python and matlab
    idx = np.argmax(np.mean(np.reshape(csi_mean_var_ratio, (ant_cnt, subcarrier_cnt)), axis=1))
    # print("idx", idx)
    csi_data_ref = np.tile(csi_data[:, idx * subcarrier_cnt:(idx + 1) * subcarrier_cnt], (1, ant_cnt))
    # print("csi_data_ref", csi_data_ref)
    
    # amplitude adjust [IndoTrack]
    csi_data_adj = np.zeros_like(csi_data)
    csi_data_ref_adj = np.zeros_like(csi_data_ref)
    alpha_sum = 0
    for jj in range(subcarrier_cnt * ant_cnt):
        amp = np.abs(csi_data[:, jj])
        alpha = np.min(amp[amp != 0])
        alpha_sum += alpha
        csi_data_adj[:, jj] = np.abs(np.abs(csi_data[:, jj]) - alpha) * np.exp(1j * np.angle(csi_data[:, jj]))
    # print("alpha_sum", alpha_sum)
    beta = 1000 * alpha_sum / (subcarrier_cnt * ant_cnt)
    # print("beta", beta)
    for jj in range(subcarrier_cnt * ant_cnt):
        csi_data_ref_adj[:, jj] = (np.abs(csi_data_ref[:, jj]) + beta) * np.exp(1j * np.angle(csi_data_ref[:, jj]))
    # print("csi_data_adj", csi_data_adj)
    # print("csi_data_ref_adj", csi_data_ref_adj)
    
    # complex conjugate multiplication and filtering
    conj_mult = csi_data_adj * np.conj(csi_data_ref_adj)
    conj_mult = np.concatenate((conj_mult[:,0:subcarrier_cnt*idx], conj_mult[:,subcarrier_cnt*(idx+1):subcarrier_cnt*ant_cnt]), axis=1)
    # print("conj_mult", conj_mult)
    for jj in range(conj_mult.shape[1]):
        # print(conj_mult[:, jj])
        conj_mult[:, jj] = lfilter(lu, ld, conj_mult[:, jj])
        # print(lu, ld, conj_mult[:, jj])
        conj_mult[:, jj] = lfilter(hu, hd, conj_mult[:, jj])
        # print(hu, hd, conj_mult[:, jj])
    
    # print("conj_mult", conj_mult)
    # PCA analysis
    pca_coef = complex_pca(conj_mult, n_components=1)[0]
    # print("pca_coef", pca_coef)
    conj_mult_pca = conj_mult @ pca_coef
    # print("conj_mult_pca", conj_mult_pca.shape)
    
    # time-frequency analysis with cwt or stft
    if method == 'cwt':
        # seems to be missing for scaled_cwt in matlab
        freq_time_prof_allfreq = pywt.cwt(conj_mult_pca, \
            scales=np.arange(1, new_samp_rate/2), wavelet='cmor')[0]
    elif method =='stft':
        time_instance = np.arange(0, new_len)
        if window_size % 2 == 0:
            window_size += 1
        cur_window = gaussian_window(window_size)
        conj_mult_pca = conj_mult_pca.squeeze()
        freq_time_prof_allfreq = tfrsp(conj_mult_pca, \
            time_instance, new_samp_rate, cur_window)[0]
        # print("freq_time_prof_allfreq", freq_time_prof_allfreq)
    
    # select concerned frequency bins
    freq_time_prof = freq_time_prof_allfreq[freq_lpf_sele,:]
    # spectrum normalization by sum for each snapshot
    freq_time_prof = np.abs(freq_time_prof) / np.sum(np.abs(freq_time_prof), axis=0)
    # print("freq_time_prof", freq_time_prof.shape, freq_time_prof)
    # print("freq_time_prof", freq_time_prof.shape) # [121,T]
    
    # frequency bin (corresponding to FFT results)
    freq_bin = np.concatenate((np.arange(0, freq_lpf_positive_max+1), np.arange(-freq_lpf_negative_min, 0)), axis=0)
    return conj_mult_pca, freq_time_prof, freq_bin

