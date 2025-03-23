import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from cyclegan_model import CycleGANModel
from Dataset import CSIDataset

def main(args):
    user2_source_dataset = CSIDataset(args.dataset_dir, \
        'cycle_source_all_csi', 'cycle_source_all_label', args.normalize_csi)
    user2_source_loader = DataLoader(user2_source_dataset, batch_size=args.batch_size, shuffle=False)
    
    cycle_source_all_csi = np.load(os.path.join(args.dataset_dir, 'cycle_source_all_csi.npy'))
    unconditional_cycle_target_all_syn_csi = np.zeros_like(cycle_source_all_csi)
    cycle_source_all_label = np.load(os.path.join(args.dataset_dir, 'cycle_source_all_label.npy'))
    unconditional_cycle_target_all_syn_label = np.zeros_like(cycle_source_all_label)
    
    # set up model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cyclegan_model = CycleGANModel(args, device)
    cyclegan_state_dict = torch.load(args.cyclegan_checkpoint_path, map_location=device)
    cyclegan_model.load_state_dict(cyclegan_state_dict)
    cyclegan_model.to(device)
    cyclegan_model.eval()
    
    for i, (src_csi_data, _) in enumerate(user2_source_loader):
        src_csi_data = src_csi_data.to(device)
        tgt_csi_data = cyclegan_model.G_A2B(src_csi_data)
        cur_batch_size = src_csi_data.shape[0]
        # the dataset read csi [bs,200,30,3] and transform to [bs,3,200,30]
        tgt_csi_data = tgt_csi_data.permute(0,2,3,1).cpu().detach().numpy()
        unconditional_cycle_target_all_syn_csi[i*args.batch_size:i*args.batch_size+cur_batch_size] = \
            tgt_csi_data
    
    np.save(os.path.join(args.syn_dataset_dir, 'unconditional_cycle_target_all_syn_csi.npy'), unconditional_cycle_target_all_syn_csi)
    np.save(os.path.join(args.syn_dataset_dir, 'unconditional_cycle_target_all_syn_label.npy'), unconditional_cycle_target_all_syn_label)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../../0_real_data_preparation/csigan_data', help='path of the dataset')
    parser.add_argument('--syn_dataset_dir', default='../../5_microbenchmark/csigan/unconditional_synthetic_data', help='path of the synthetic dataset')
    parser.add_argument('--epochs',type=int, default=1000, help='# of epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='# images in batch')
    
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discri filters in first conv layer')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--normalize_csi', action='store_true', help='scale csi to [-1, 1]')
    parser.add_argument('--cyclegan_checkpoint_path', default='cyclegan_checkpoints/shuffled_trained_ckpt/epoch4_l1trained0.7537_l1all0.7887.pth', help='path of the cyclegan checkpoint')
    args = parser.parse_args()
    
    main(args)
