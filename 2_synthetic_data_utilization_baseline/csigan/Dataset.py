from torch.utils.data import Dataset
import os
import numpy as np
import torch

# read csi [bs,200,30,3] and transform to [bs,90,200] or [bs,3,200,30]
class SignFiCSIDataset(Dataset):
    def __init__(self, root_dir, csi_fname, label_fname, normalize_csi=True,
                 transform=None, return_idx=False, reshape_2dim=False, add_half=0):
        super(SignFiCSIDataset, self).__init__()
        self.root_dir = root_dir
        self.csi_fname = csi_fname
        self.label_fname = label_fname
        self.normalize_csi = normalize_csi
        self.transform = transform
        self.return_idx = return_idx
        self.reshape_2dim = reshape_2dim
        self.add_half = add_half
        
        csi_data = np.load(os.path.join(root_dir, f'{csi_fname}.npy'))
        label_data = np.load(os.path.join(root_dir, f'{label_fname}.npy'))
        if self.reshape_2dim:
            csi_data = csi_data.reshape(-1, 200, 30*3)
            self.csi_data = np.transpose(csi_data, (0,2,1))
        else:
            self.csi_data = np.transpose(csi_data, (0,3,1,2))
        self.label_data = label_data
        
        self.selected_indices = None
    
    def __len__(self):
        if self.selected_indices is not None:
            return len(self.selected_indices)
        else:
            return len(self.label_data)
    
    def set_index(self, indices):
        if len(indices) == 0:
            indices = [0]
        self.selected_indices = indices
    
    def init_index(self):
        self.selected_indices = None
    
    def __getitem__(self, idx):
        ori_idx = idx
        if self.selected_indices is not None:
            idx = self.selected_indices[idx]
            
        csi = self.csi_data[idx]
        label = self.label_data[idx].astype(np.int64)
            
        if self.normalize_csi:
            csi = (csi - np.mean(csi)) / np.std(csi)
        csi = torch.from_numpy(csi).float()
        label = torch.tensor(label).long()
        if self.transform is not None:
            csi = self.transform(csi)
        if self.add_half > 0:
            label += self.add_half
        if self.return_idx:
            idx = torch.tensor(idx).long()
            return csi, label, idx
        else:
            return csi, label