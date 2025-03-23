import numpy as np
import os
import torch
from torch.utils.data import Dataset

class RealDFSDataset(Dataset):
    """
    Dataset composed of real-world DFS
    """
    def __init__(self, data_dir, fname_label_dir, part, rx_list, normalize_dfs=True, return_ids=False):
        super(RealDFSDataset, self).__init__()
        self.data_dir = data_dir
        self.fname_label_dir = fname_label_dir
        self.part = part
        self.rx_list = rx_list

        self.normalize_dfs = normalize_dfs
        self.return_ids = return_ids
        self.records = np.load(os.path.join(fname_label_dir, f'{part}_filename.npy'))
        self.labels = np.load(os.path.join(fname_label_dir, f'{part}_label.npy'))
        
        target_list = []
        for cur_label in self.labels:
            for cur_rx in self.rx_list:
                target_list.append(cur_label-1)
        self.targets = torch.tensor(target_list)
        
    def __len__(self):
        return len(self.records) * len(self.rx_list)
    
    def __getitem__(self, idx):
        file_idx = idx // len(self.rx_list)
        rx_idx = idx % len(self.rx_list)
        record = self.records[file_idx]
        label = self.labels[file_idx] - 1 
        target = self.targets[idx]
        assert label == target
        
        data_path = os.path.join(self.data_dir, f'{record}-r{self.rx_list[rx_idx]}.npz')
        doppler_spectrum = np.load(data_path, allow_pickle=True)['doppler_spectrum']
        
        data = doppler_spectrum
        if self.normalize_dfs:
            data = (data - np.mean(data)) / np.std(data)
        data = torch.from_numpy(data).float()
        label = torch.tensor(label).long()
        idx = torch.tensor(idx).long()
        if self.return_ids:
            return data, label, idx
        return data, label



class SynDFSDataset(Dataset):
    """
    Dataset composed of synthetic DFS
    """
    def __init__(self, data_dir, fname_label_dir, part, rx_list, normalize_dfs=True, return_ids=False):
        super(SynDFSDataset, self).__init__()
        self.data_dir = data_dir
        self.fname_label_dir = fname_label_dir
        self.part = part
        self.rx_list = rx_list

        self.normalize_dfs = normalize_dfs
        self.return_ids = return_ids
        self.records = np.load(os.path.join(fname_label_dir, f'{part}_filename.npy'))
        self.labels = np.load(os.path.join(fname_label_dir, f'{part}_label.npy'))
        
        target_list = []
        for cur_label in self.labels:
            for cur_rx in self.rx_list:
                target_list.append(cur_label-1)
        self.targets = torch.tensor(target_list)
    
    def __len__(self):
        return len(self.records) * len(self.rx_list)
    
    def __getitem__(self, idx):
        file_idx = idx // len(self.rx_list)
        rx_idx = idx % len(self.rx_list)
        record = self.records[file_idx]
        label = self.labels[file_idx] - 1 
        target = self.targets[idx]
        assert label == target
        
        data_path = os.path.join(self.data_dir, f'{record}-r{self.rx_list[rx_idx]}.npz')
        doppler_spectrum = np.load(data_path, allow_pickle=True)['pred_doppler_spectrum']
        
        data = doppler_spectrum
        if self.normalize_dfs:
            data = (data - np.mean(data)) / np.std(data)
        data = torch.from_numpy(data).float()
        label = torch.tensor(label).long()
        idx = torch.tensor(idx).long()
        if self.return_ids:
            return data, label, idx
        return data, label
    

class ConcatDataset(Dataset):
    """
    Dataset composed of concatenated datasets
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        cur_element = self.datasets[dataset_idx][sample_idx]
        if self.datasets[dataset_idx].return_ids:
            new_element = cur_element[:-1] + (idx,)
            return new_element
        else:
            return cur_element

class SubsetDataset(Dataset):
    """
    Dataset composed of a subset of another dataset
    """
    def __init__(self, dataset, indices):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.indices = indices
        self.return_ids = dataset.return_ids
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        cur_idx = self.indices[idx]
        cur_element = self.dataset[cur_idx]
        if self.dataset.return_ids:
            new_element = cur_element[:-1] + (idx,)
            return new_element
        else:
            return cur_element
        