import os
from tfdiff.Dataset import WiFiDataset
from tfdiff.wifi_model import tidiff_WiFi_unconditional
from tfdiff.diffusion import SignalDiffusion
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
from utils import get_current_time
import torch
import torch.nn as nn
import argparse
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from rfdiff_utils import eval_ssim

CURRENT_TIME: str = get_current_time()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model and dataset
    if args.task_id == 0:
        model = tidiff_WiFi_unconditional(args).to(device)
        diffusion = SignalDiffusion(args)
        train_dataset = WiFiDataset(args.data_dir, args.sample_rate, args.cond_dim)
        val_dataset = WiFiDataset(args.val_data_dir, args.sample_rate, args.cond_dim)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{args.log_dir}/{CURRENT_TIME}.txt", mode="w", encoding="utf-8")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(len(train_dataset))
    
    # set up tensorboard writer
    step_name = f'steps{args.max_step}'
    checkpoint_dir = os.path.join(args.checkpoint_root_dir, 
        f'unconditional-{step_name}-{CURRENT_TIME}')
    writer = SummaryWriter(log_dir=checkpoint_dir)
    logger.info(f"Writing tensorboard logs to {checkpoint_dir}")
    
    # set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    logger.info(optimizer)
    
    # training loop
    loss_fn = nn.MSELoss()
    for epoch in range(args.num_epochs):
        # validate at the end of each epoch
        model.eval()
        native_ssim_list = []
        fast_ssim_list = []
        for batch_idx, (data, cond) in enumerate(tqdm(val_loader)):
            data, cond = data.to(device), cond.to(device)
            cur_batch_size = data.shape[0]
            # test different sampling methods
            native_pred = diffusion.native_sampling(model, data, cond, device).detach()
            torch.cuda.empty_cache()
            fast_pred = diffusion.fast_sampling(model, cond, device).detach()
            torch.cuda.empty_cache()
            data_samples = [torch.view_as_complex(sample) for sample in torch.split(data, 1, dim=0)] # [B, [1, N, S]]
            native_samples = [torch.view_as_complex(sample) for sample in torch.split(native_pred, 1, dim=0)] # [B, [1, N, S]]
            fast_samples = [torch.view_as_complex(sample) for sample in torch.split(fast_pred, 1, dim=0)] # [B, [1, N, S]]
            for i in range(cur_batch_size):
                native_ssim = eval_ssim(native_samples[i], data_samples[i], args.sample_rate, args.input_dim, device=device)
                fast_ssim = eval_ssim(fast_samples[i], data_samples[i], args.sample_rate, args.input_dim, device=device)
                native_ssim_list.append(native_ssim.item())
                fast_ssim_list.append(fast_ssim.item())
        native_ssim_mean = np.mean(np.array(native_ssim_list))
        fast_ssim_mean = np.mean(np.array(fast_ssim_list))
        logger.info(f"Epoch: {epoch}, Native SSIM: {native_ssim_mean}, Fast SSIM: {fast_ssim_mean}")
        writer.add_scalar('val/native_ssim', native_ssim, epoch)
        writer.add_scalar('val/fast_ssim', fast_ssim, epoch)
        
        model.train()
        all_loss = 0
        for batch_idx, (data, cond) in enumerate(tqdm(train_loader)):
            data, cond = data.to(device), cond.to(device)
            cur_batch_size = data.shape[0]
            # random diffusion step
            t = torch.randint(0, diffusion.max_step, [cur_batch_size], dtype=torch.int64)
            degrade_data = diffusion.degrade_fn(data, t)
            predicted = model(degrade_data, t, cond)
            if args.task_id == 3:
                data = data.reshape(-1, args.sample_rate, 1, 2)
            
            optimizer.zero_grad()
            recon_loss = loss_fn(data, predicted)
            all_loss += cur_batch_size * recon_loss.item()
            recon_loss.backward()
            
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm or 1e9)
            optimizer.step()
            scheduler.step()
            
            if batch_idx % args.log_interval == 0:
                logger.info(f"Epoch: {epoch}, Batch: {batch_idx}, Recon Loss: {recon_loss.item()}, Grad Norm: {grad_norm}")
                writer.add_scalar('train/recon_loss', recon_loss.item(), epoch * len(train_loader) + batch_idx)

        if epoch % args.save_epoch == 0:
            all_loss /= len(train_loader.dataset)
            save_basename = f'{epoch}-loss{all_loss:.4f}.pth'
            save_path = os.path.join(checkpoint_dir, save_basename)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved model to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0, help='0/1/2/3 for WiFi/FMCW/MIMO/EEG')
    # logging
    parser.add_argument('--log_dir', type=str, default='rfdiffusion_logs', help='path to log directory')
    parser.add_argument('--checkpoint_root_dir', type=str, default='rfdiffusion_checkpoints', help='path to checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='interval to log')
    parser.add_argument('--save_epoch', type=int, default=1, help='interval to save model')
    # data
    parser.add_argument('--data_dir', default=['../../0_real_data_preparation/cond_mat_CSI/20181130','../../0_real_data_preparation/cond_mat_CSI/20181204','../../0_real_data_preparation/cond_mat_CSI/20181209','../../0_real_data_preparation/cond_mat_CSI/20181211'], help='path to data directory', 
        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--val_data_dir', default=['../../0_real_data_preparation/cond_mat_CSI/20181204'], help='path to validation data directory', 
        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--sample_rate', type=int, default=512)
    parser.add_argument('--input_dim', type=int, default=90)
    parser.add_argument('--extra_dim', default=[90], 
        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--cond_dim', type=int, default=6, help='number of conditions')
    # training params
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
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
    
    args = parser.parse_args()
    args.blur_schedule = ((args.gaussian_std**2) * np.ones(args.max_step)).tolist()
    args.noise_schedule = np.linspace(args.noise_start, args.noise_end, args.max_step).tolist()
    
    main(args)
    