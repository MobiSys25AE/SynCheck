import numpy as np
import struct
from scipy.signal import butter, lfilter, get_window
from scipy.fft import fft, fftshift
import pywt
import os
import torch
import torch.nn.functional as F

# unpack the binary data to store CSI
def read_bfee(in_bytes):
    # extract fields from binary data
    timestamp_low = struct.unpack('<I', in_bytes[0:4])[0]
    bfee_count = struct.unpack('<H', in_bytes[4:6])[0]
    Nrx = in_bytes[8]
    Ntx = in_bytes[9]
    rssi_a = in_bytes[10]
    rssi_b = in_bytes[11]
    rssi_c = in_bytes[12]
    # signed char noise
    noise = struct.unpack('<b', in_bytes[13:14])[0]
    agc = in_bytes[14]
    antenna_sel = in_bytes[15]
    length = struct.unpack('<H', in_bytes[16:18])[0]
    fake_rate_n_flags = struct.unpack('<H', in_bytes[18:20])[0]
    calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 7) // 8
    
    # Check that length matches what it should
    if length != calc_len:
        raise ValueError("Wrong beamforming matrix size.")
    
    # Processing CSI data
    csi = np.zeros((Ntx, Nrx, 30), dtype=np.complex64)
    index = 0
    payload = in_bytes[20:]
    for i in range(30):
        index += 3
        for j in range(Nrx * Ntx):
            idx = index // 8
            remainder = index % 8
            tmp_real = ((payload[idx] >> remainder) | (payload[idx + 1] << (8 - remainder))) & 0xFF
            tmp_imag = ((payload[idx + 1] >> remainder) | (payload[idx + 2] << (8 - remainder))) & 0xFF
            # make tmp_real and tmp_imag as signed 8-bit integers
            tmp_real = struct.unpack('<b', bytes([tmp_real]))[0]
            tmp_imag = struct.unpack('<b', bytes([tmp_imag]))[0]
            csi[j%Ntx, j//Ntx, i] = tmp_real + 1j * tmp_imag
            
            index += 16
    
    perm = [(antenna_sel >> (2 * i)) & 0x03 for i in range(3)]
    
    result = {
        'timestamp_low': timestamp_low, 'bfee_count': bfee_count,
        'Nrx': Nrx, 'Ntx': Ntx,
        'rssi_a': rssi_a, 'rssi_b': rssi_b, 'rssi_c': rssi_c,
        'noise': noise, 'agc': agc, 'perm': perm,
        'rate': fake_rate_n_flags,
        'csi': csi
    }
    
    return result


def read_bf_file(filename):
    try:
        with open(filename, 'rb') as f:
            # Get file length
            f.seek(0, 2)    # Move to end of file
            file_length = f.tell()
            f.seek(0, 0)    # Move back to beginning of file
            
            ret = [None] * (file_length // 95)  # 1x1 CSI is 95 bytes big
            cur = 0                     # Current offset into file
            count = 0                   # Number of records output
            broken_perm = 0             # Flag marking whether we've encountered a broken CSI
            triangle = [0, 1, 3]        # What perm should sum to for 1,2,3 antennas
            
            # 3 bytes: 2 byte size field and 1 byte code
            while cur < (file_length-3):
                # read size and code
                size_field = f.read(2)
                field_len = struct.unpack('>H', size_field)[0]
                code = ord(f.read(1))
                cur += 3
                
                if code == 187: # Beamforming or phy data
                    bytes = f.read(field_len-1)
                    cur += field_len-1
                    if len(bytes) != (field_len-1):
                        break
                else:   # skip all other info
                    f.seek(field_len-1, 1)
                    cur += field_len-1
                    continue
                
                if code == 187:
                    count += 1
                    result = read_bfee(bytes)
                    ret[count-1] = result
                    perm = result['perm']
                    Nrx = result['Nrx']
                    if Nrx == 1:    # No permutation need for only 1 antenna
                        continue
                    if sum(perm) != triangle[Nrx-1]:
                        print(f'WARN ONCE: Found CSI ({filename}) with Nrx={Nrx} and invalid perm={perm}, triangle={triangle}')
                    else:
                        # reorder the csi according to perm
                        old_csi = result['csi'].copy()
                        for ri in range(Nrx):
                            ret[count-1]['csi'][:,perm[ri],:] = old_csi[:,ri,:]
                        
            ret = ret[:count]
            return ret
        
    except IOError as e:
        print(f"Couldn't open file {filename}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def dbinv(x):
    return 10 ** (x / 10)

def db(x):
    return 10 * np.log10(x)

def get_total_rss(csi_st):
    # Initialize the RSSI magnitude sum
    rssi_mag = 0
    if csi_st['rssi_a'] != 0:
        rssi_mag += dbinv(csi_st['rssi_a'])
    if csi_st['rssi_b'] != 0:
        rssi_mag += dbinv(csi_st['rssi_b'])
    if csi_st['rssi_c'] != 0:
        rssi_mag += dbinv(csi_st['rssi_c'])
    
    # convert the summed magnitude to dB, adjust by constant and AGC
    total_rss = db(rssi_mag) - 44 - csi_st['agc']
    return total_rss


def get_scaled_csi(csi_st):
    # Extract CSI from the struct
    csi = csi_st['csi']
    # Calculate the scale factor between normalized CSI and RSSI (mW)
    csi_sq = np.abs(csi) ** 2
    csi_pwr = np.sum(csi_sq)
    rssi_pwr = dbinv(get_total_rss(csi_st))
    # Scale CSI -> Signal power: rssi_pwr / (mean of csi_pwr)
    scale = rssi_pwr / (csi_pwr / 30)
    
    # Thermal noise might be undefined if the trace was capture in monitor mode. If so, set it to -92
    if csi_st['noise'] == -127:
        noise_db = -92
    else:
        noise_db = csi_st['noise']
    thermal_noise_pwr = dbinv(noise_db)
    
    # Quantization error: the coefficients in the matrices are 8-bit signed numbers, max 127/-128 to min 0/1. Given that Intel only uses a 6-bit ADC, I expect every entry to be off by about +/-1 (total across real & complex parts) per entry. 
    # The total power is than 1^2 = 1 per entry, and there are Nrx*Ntx entries per carrier. We only want one carrier's worth of error, since we only computed one carrier's worth of signal above. 
    quant_error_pwr = scale * (csi_st['Nrx'] * csi_st['Ntx'])
    # total noise and error power
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr
    
    # Ret now has units of sqrt(SNR) just like H in textbooks
    ret = csi * np.sqrt(scale / total_noise_pwr)
    if csi_st['Ntx'] == 2:
        ret *= np.sqrt(2)
    elif csi_st['Ntx'] == 3:
        # This should be sqrt(3)~4.77 dB. But 4.5 dB is how Intel (and some other chip makers) approximate a factor of 3. You may need to change this if your card does the right thing. 
        ret *= np.sqrt(dbinv(4.5))
    
    return ret

def csi_get_all(filename):
    csi_trace = read_bf_file(filename)
    packet_cnt = len(csi_trace)
    timestamp = np.zeros((packet_cnt, 1))
    cfr_array = np.zeros((packet_cnt, 90), dtype=np.complex64)
    
    for i in range(packet_cnt):
        csi_entry = csi_trace[i]
        # shape [3,30]
        csi_all = get_scaled_csi(csi_entry).squeeze()
        # reshape into [1,90]
        cfr_array[i,:] = csi_all.reshape(1,90)
        timestamp[i] = csi_entry['timestamp_low']
        
    return cfr_array, timestamp


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
    
    # compute the power spectrum
    tfr = np.abs(tfr) ** 2
    # frequency vector
    if N % 2 == 0:
        f = np.arange(0,N/2+1).tolist() + np.arange(-N/2,0).tolist()
    else:
        f = np.arange(0, (N+1)/2).tolist() + np.arange(-(N-1)/2,0).tolist()
    
    return tfr, t_indices, np.array(f) / N


# single antenna CSI data: shape (T, f) (1433, 90), ant_cnt = 3
# reshaped into 512, and change samp_rate accordingly
def get_doppler_spectrum(csi_data, method, window_size=251, ant_cnt=3, subcarrier_cnt=30):
    # set up parameters
    ori_packet_cnt = csi_data.shape[0]
    new_len = 512
    ori_samp_rate = 1000
    new_samp_rate = int(new_len / (ori_packet_cnt / ori_samp_rate))
    new_samp_rate = (new_samp_rate // 2) * 2
    
    # normalized the CSI data: this is used in RF-Diffusion
    data = torch.from_numpy(csi_data).to(torch.complex64)
    data = torch.view_as_real(data).permute(1, 2, 0)
    down_sample = F.interpolate(data, new_len, mode='nearest-exact')
    norm_data = (down_sample - down_sample.mean()) / down_sample.std()
    norm_data = norm_data.permute(2, 0, 1).contiguous()
    norm_array = torch.view_as_complex(norm_data)
    print(new_samp_rate, csi_data.shape, norm_array.shape, norm_array.dtype, norm_data.mean())
    csi_data = norm_array.numpy()
    
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
    return csi_data, conj_mult_pca, freq_time_prof, freq_bin

skipped_file_list = []
def process_file(cur_user_dir, cur_file, new_len, cur_dst_dir):
    cur_file_path = os.path.join(cur_user_dir, cur_file)
    cfr_array, timestamp = csi_get_all(cur_file_path)
    if cfr_array.shape[0] < new_len:
        print(f"Skipping {cur_file_path} with {cfr_array.shape[0]} packets.")
        skipped_file_list.append(cur_file_path)
    
    cur_window_size = new_len//4+1
    csi_data, conj_mult_pca, freq_time_prof, freq_bin = \
        get_doppler_spectrum(cfr_array, method='stft', window_size=cur_window_size)
    # save freq_time_prof into .npz files
    np.savez(os.path.join(cur_dst_dir, cur_file[:-4]), \
        ori_packet_cnt=cfr_array.shape[0], reshaped_csi=csi_data, 
        doppler_spectrum=freq_time_prof, conj_mult_pca=conj_mult_pca)
    

from concurrent.futures import ThreadPoolExecutor, as_completed
if __name__ == '__main__':
    csi_src_dir = 'your_src_csi_data_dir'
    dfs_dst_dir = 'your_dst_dfs_data_dir'
    
    os.makedirs(dfs_dst_dir, exist_ok=True)
    # all_dates = [name for name in os.listdir(csi_src_dir) if os.path.isdir(os.path.join(csi_src_dir, name))]
    all_dates = ['20181130', '20181204', '20181209', '20181211']
    
    new_len = 512
    with ThreadPoolExecutor() as executor:
        futures = []
        for cur_date in all_dates:
            cur_dir = os.path.join(csi_src_dir, cur_date)
            all_users = [name for name in os.listdir(cur_dir) if os.path.isdir(os.path.join(cur_dir, name))]
            for cur_user in all_users:
                cur_user_dir = os.path.join(cur_dir, cur_user)
                all_files = [name for name in os.listdir(cur_user_dir) if os.path.isfile(os.path.join(cur_user_dir, name)) and name.endswith('.dat')]
                cur_dst_dir = os.path.join(dfs_dst_dir, cur_date, cur_user)
                os.makedirs(cur_dst_dir, exist_ok=True)
                for cur_file in all_files:
                    future = executor.submit(process_file, cur_user_dir, cur_file, new_len, cur_dst_dir)
                    futures.append(future)
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
