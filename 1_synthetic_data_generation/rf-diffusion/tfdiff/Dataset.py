import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import scipy.io as scio
import torch.nn.functional as F

class WiFiDataset(Dataset):
    def __init__(self, paths, sample_rate, cond_length):
        super().__init__()
        self.sample_rate = sample_rate
        self.cond_length = cond_length
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/user*.mat', recursive=True)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        cur_filename = self.filenames[idx]
        cur_sample = scio.loadmat(cur_filename,verify_compressed_data_integrity=False)
        # [T, 90]
        cur_data = torch.from_numpy(cur_sample['feature']).to(torch.complex64)
        cur_cond = torch.from_numpy(cur_sample['cond']).to(torch.complex64)
        if cur_data.shape[0] < self.sample_rate:
            return self.__getitem__((idx+1) % len(self.filenames))
        else:
            cur_data = torch.view_as_real(cur_data)
            cur_data = cur_data.permute(1,2,0)
            down_sample = F.interpolate(cur_data, self.sample_rate, mode='nearest-exact')
            down_sample = down_sample.permute(2, 0, 1).squeeze()
            norm_data = (down_sample - down_sample.mean()) / down_sample.std()
            cur_cond = cur_cond.squeeze(0)
            
        return norm_data, torch.view_as_real(cur_cond)


class DFSCSICondDataset(Dataset):
    def __init__(self, data_dir, cond_dir, fname_label_dir,  
                part, rx_list, normalize_dfs):
        super(DFSCSICondDataset, self).__init__()
        self.data_dir = data_dir
        self.cond_dir = cond_dir
        self.fname_label_dir = fname_label_dir
        self.part = part        
        self.rx_list = rx_list
        self.normalize_dfs = normalize_dfs
        # CSI already normed
        
        self.records = np.load(os.path.join(fname_label_dir, f'{part}_filename.npy'))
        self.labels = np.load(os.path.join(fname_label_dir, f'{part}_label.npy'))
        self.data_cnt = len(self.records) * len(self.rx_list)
        
    def __len__(self):
        return self.data_cnt
    
    def __getitem__(self, idx):
        file_idx = idx // len(self.rx_list)
        rx_idx = idx % len(self.rx_list)
        record = self.records[file_idx]
        label = self.labels[file_idx] - 1
        data_path = os.path.join(self.data_dir, f'{record}-r{self.rx_list[rx_idx]}.npz')
        cond_path = os.path.join(self.cond_dir, f'{record}-r{self.rx_list[rx_idx]}.mat')
        if not os.path.exists(data_path):
            print(data_path, ' not exists')
            return self.__getitem__((idx + 1) % self.real_cnt)
        
        # [512,90]
        csi_data = np.load(data_path, allow_pickle=True)['reshaped_csi']
        csi_data = torch.view_as_real(torch.from_numpy(csi_data).to(torch.complex64))
        # [121,90]
        doppler_spectrum = np.load(data_path, allow_pickle=True)['doppler_spectrum']
        cond = scio.loadmat(cond_path, verify_compressed_data_integrity=False)['cond']
        cond = torch.from_numpy(cond).to(torch.complex64).squeeze(0)
        cond = torch.view_as_real(cond)
        ori_packet_cnt = np.load(data_path, allow_pickle=True)['ori_packet_cnt']
        
        if self.normalize_dfs:
            doppler_spectrum = (doppler_spectrum - np.mean(doppler_spectrum)) / np.std(doppler_spectrum)
        doppler_spectrum = torch.from_numpy(doppler_spectrum).float()
        label = torch.tensor(label).long()
        ori_packet_cnt = torch.tensor(ori_packet_cnt).long()    
        idx = torch.tensor(idx).long()
        
        
        return csi_data, doppler_spectrum, cond, label, ori_packet_cnt, idx
        



