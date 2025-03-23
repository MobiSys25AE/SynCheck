from torch.utils.data import Dataset
import os
import numpy as np
import torch

# read csi [bs,200,30,3] and transform to [bs,3,200,30]
class CSIDataset(Dataset):
    def __init__(self, root_dir, csi_fname, label_fname, normalize_csi=True, 
                 is_syn=False, category_cnt=50, return_ids=False):
        super(CSIDataset, self).__init__()
        self.root_dir = root_dir
        self.csi_fname = csi_fname
        self.label_fname = label_fname
        self.normalize_csi = normalize_csi
        self.is_syn = is_syn
        self.category_cnt = category_cnt
        self.return_ids = return_ids
        
        csi_data = np.load(os.path.join(root_dir, f'{csi_fname}.npy'))
        label_data = np.load(os.path.join(root_dir, f'{label_fname}.npy'))
        self.csi_data = np.transpose(csi_data, (0,3,1,2))
        self.label_data = label_data
    
    def __len__(self):
        return len(self.label_data)
    
    def __getitem__(self, idx):
        csi = self.csi_data[idx]
        label = self.label_data[idx].astype(np.int64)
        if self.is_syn:
            label += self.category_cnt
            
        if self.normalize_csi:
            csi = (csi - np.mean(csi)) / np.std(csi)
        csi = torch.from_numpy(csi).float()
        label = torch.tensor(label).long()
        if self.return_ids:
            idx = torch.tensor(idx).long()
            return csi, label, idx
        return csi, label


class CycleDataset(Dataset):
    def __init__(self, source_dataset, target_dataset, shuffle_pair=False):
        super(CycleDataset, self).__init__()
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.shuffle_pair = shuffle_pair
    
    def __len__(self):
        return len(self.source_dataset)
    
    def __getitem__(self, idx):
        source_csi, source_label = self.source_dataset[idx]
        if not self.shuffle_pair:
            target_csi, target_label = self.target_dataset[idx]
        else:
            target_idx = np.random.randint(len(self.target_dataset))
            target_csi, target_label = self.target_dataset[target_idx]
        return source_csi, source_label, target_csi, target_label

