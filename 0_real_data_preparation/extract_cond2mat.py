import numpy as np
import struct
import os

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

import os
import numpy as np
from scipy.io import savemat
from concurrent.futures import ThreadPoolExecutor, as_completed

date2roomid_dict = {
    '20181109': 1, '20181112': 1, '20181115': 1, '20181116': 1, 
    '20181117': 2, '20181118': 2, 
    '20181121': 1, '20181127': 2, '20181128': 2, '20181130': 1, 
    '20181204': 2, '20181205': 2, '20181208': 2, '20181209': 2, 
    '20181211': 3
}

# all_dates = date2roomid_dict.keys()
all_dates = ['20181130', '20181204', '20181209', '20181211']
root_dir = 'you_src_csi_dir'
dst_dir = 'cond_mat_CSI'
os.makedirs(dst_dir, exist_ok=True)

def process_file(cur_date, cur_roomid, cur_user, cur_dat_file):
    cur_user_dir = os.path.join(root_dir, cur_date, cur_user)
    cur_dat_path = os.path.join(cur_user_dir, cur_dat_file)
    
    # Assume csi_get_all is a defined function that returns a tuple
    cfr_array, _ = csi_get_all(cur_dat_path)
    
    cond_list = [cur_roomid]
    fname_list = cur_dat_file.split('-')
    mid_conds = [int(c) for c in fname_list[1:4]]
    cond_list += mid_conds
    rx_part = fname_list[-1]
    rx_part = rx_part.split('.')[0]
    cond_list.append(int(rx_part[1:]))
    user_part = fname_list[0]
    cond_list.append(int(user_part[4:]))
    
    print(cur_dat_file, cond_list)

    # Save CSI data and conditions into .mat file
    data = {
        'feature': cfr_array, 
        'cond': np.array(cond_list)
    }
    cur_dst_dir = os.path.join(dst_dir, cur_date, cur_user)
    os.makedirs(cur_dst_dir, exist_ok=True)
    dst_path = os.path.join(cur_dst_dir, cur_dat_file.split('.')[0] + '.mat')
    savemat(dst_path, data)

def process_user(cur_date, cur_roomid, cur_user):
    cur_user_dir = os.path.join(root_dir, cur_date, cur_user)
    cur_dat_files = os.listdir(cur_user_dir)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, cur_date, cur_roomid, cur_user, cur_dat_file)
                   for cur_dat_file in cur_dat_files]
        for future in as_completed(futures):
            future.result()

def main():
    with ThreadPoolExecutor() as executor:
        futures = []
        for cur_date in all_dates:
            cur_roomid = date2roomid_dict[cur_date]
            cur_dir = os.path.join(root_dir, cur_date)
            cur_users = os.listdir(cur_dir)
            for cur_user in cur_users:
                futures.append(executor.submit(process_user, cur_date, cur_roomid, cur_user))
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    main()



