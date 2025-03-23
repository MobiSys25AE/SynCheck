import os
import numpy as np
import torch
from tfdiff.Dataset import DFSCSICondDataset
from torch.utils.data import DataLoader
from tfdiff.diffusion import SignalDiffusion
from tfdiff.wifi_model import tidiff_WiFi
from rfdiff_utils import eval_ssim, real_ssim
import argparse
import torch.nn.functional as F
from CSI2DFS import get_doppler_spectrum
from tqdm import tqdm

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.syn_save_dir, exist_ok=True)
    model = tidiff_WiFi(args).to(device)
    diffusion = SignalDiffusion(args)
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Generate synthetic data
    real_dataset = DFSCSICondDataset(args.data_dir, args.cond_dir, \
        args.fname_label_dir, "train", args.train_rxs, args.normalize_dfs)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)
    syn_filename_list = []
    syn_label_list = []
    for ei in range(5):
        for batch_idx, batch_data in tqdm(enumerate(real_loader)):
            csi_data, doppler_spectrum, cond, label, ori_packet_cnt, idx = batch_data
            csi_data, doppler_spectrum, cond, label = csi_data.to(device), doppler_spectrum.to(device), cond.to(device), label.to(device)
            cur_batch_size = csi_data.shape[0]
            with torch.no_grad():
                native_pred = diffusion.native_sampling(model, csi_data, cond, device).detach()
                torch.cuda.empty_cache()
                # fast_pred = diffusion.fast_sampling(model, cond, device).detach()
                # torch.cuda.empty_cache()
                data_samples = [torch.view_as_complex(sample) for sample in torch.split(csi_data, 1, dim=0)] # [B, [1, N, S]]
                native_samples = [torch.view_as_complex(sample) for sample in torch.split(native_pred, 1, dim=0)] # [B, [1, N, S]]
                # fast_samples = [torch.view_as_complex(sample) for sample in torch.split(fast_pred, 1, dim=0)] # [B, [1, N, S]]
            
            # convert the synthetic data to DFS
            for si in range(cur_batch_size):
                cur_idx = idx[si].item()
                native_ssim = eval_ssim(native_samples[si], data_samples[si], args.sample_rate, args.input_dim, device=device).item()
                # fast_ssim = eval_ssim(fast_samples[si], data_samples[si], args.sample_rate, args.input_dim, device=device).item()
                
                native_dfs = get_doppler_spectrum(native_samples[si].squeeze().cpu().numpy(), \
                        ori_packet_cnt[si].item(), 'stft')[1]
            
                # save output csi and doppler spectrum
                cur_file_idx = cur_idx // (len(args.train_rxs))
                cur_rx_idx = cur_idx % (len(args.train_rxs))
                cur_filename = real_dataset.records[cur_file_idx]
                cur_label = label[si].item() + 1
                new_filename = f'{cur_filename}-{ei}'
                # save synthetic data information
                if cur_rx_idx == 0:
                    syn_filename_list.append(new_filename)
                    syn_label_list.append(cur_label)
                save_path = os.path.join(args.syn_save_dir, \
                    f'{new_filename}-r{args.train_rxs[cur_rx_idx]}.npz')
                subdir = os.path.dirname(save_path)
                os.makedirs(subdir, exist_ok=True)
                # CSI is normalizated, DFS is not
                np.savez(save_path, pred_csi=native_samples[si].squeeze().cpu().numpy(), \
                    pred_doppler_spectrum=native_dfs)
    
    syn_filename_list = np.array(syn_filename_list)
    syn_label_list = np.array(syn_label_list)
    np.save(os.path.join(args.syn_save_dir, 'train_filename.npy'), \
        syn_filename_list)
    np.save(os.path.join(args.syn_save_dir, 'train_label.npy'), \
        syn_label_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0, help='0/1/2/3 for WiFi/FMCW/MIMO/EEG')
    # data
    parser.add_argument('--data_dir', default='../../0_real_data_preparation/real_dfs_data', help='path to data directory')
    parser.add_argument('--cond_dir', default='../../0_real_data_preparation/cond_mat_CSI')
    parser.add_argument('--fname_label_dir', default='../../0_real_data_preparation/real_fname_label', help='path to fname_label directory')
    parser.add_argument('--train_rxs', default=[1,3,5], help='selected rx', 
        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--sample_rate', type=int, default=512)
    parser.add_argument('--input_dim', type=int, default=90)
    parser.add_argument('--extra_dim', default=[90], 
        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--cond_dim', type=int, default=6, help='number of conditions')
    parser.add_argument('--normalize_dfs', action='store_true', help='Whether to normalize the DFS data.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    # model params
    parser.add_argument('--embed_dim', type=int, default=256, help='embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--num_block', type=int, default=32, help='number of blocks')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--max_grad_norm', type=float, default=None)
    # diffusion params
    parser.add_argument('--max_step', type=int, default=100, help='maximum number of diffusion steps')
    parser.add_argument('--gaussian_std', type=float, default=1e-5, help='std of gaussian blur applied to the spectrogram')
    parser.add_argument('--noise_start', type=float, default=1e-4, help='noise level at the beginning of diffusion')
    parser.add_argument('--noise_end', type=float, default=0.003, help='noise level at the end of diffusion')
    # saved restoremodel ckpt
    parser.add_argument('--ckpt_path', type=str, default='rfdiffusion_checkpoints/trained_ckpt/5-loss0.0099.pth', help='path to saved model ckpt')
    # synthetic data save directory
    parser.add_argument('--syn_save_dir', type=str, default='../../5_microbenchmark/rf-diffusion/syn_more_data_native', help='path to synthetic data directory')
    
    args = parser.parse_args()
    args.blur_schedule = ((args.gaussian_std**2) * np.ones(args.max_step)).tolist()
    args.noise_schedule = np.linspace(args.noise_start, args.noise_end, args.max_step).tolist()
    os.makedirs(args.syn_save_dir, exist_ok=True)
    
    main(args)