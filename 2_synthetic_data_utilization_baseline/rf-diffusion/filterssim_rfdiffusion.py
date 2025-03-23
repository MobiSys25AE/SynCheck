import sys
import os
from resnet1d import resnet18, resnet34, resnet50, resnet101, resnet152
from Dataset import RealDFSDataset, SynDFSDataset, SubsetDataset, ConcatDataset
from utils import get_current_time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import argparse
import copy

def real_ssim(array1, array2):
    # Calculate mean, variance, and covariance
    # Constants for SSIM calculation
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    ux = array1.mean()
    uy = array2.mean()
    var_x = array1.var()
    var_y = array2.var()
    cov_xy = np.cov(array1.flatten(), array2.flatten())[0, 1]

    # Calculate SSIM components
    A1 = 2 * ux * uy + C1
    A2 = 2 * cov_xy + C2
    B1 = ux ** 2 + uy ** 2 + C1
    B2 = var_x + var_y + C2

    # Calculate SSIM index
    ssim_index = (A1 * A2) / (B1 * B2)
    return ssim_index

CURRENT_TIME: str = get_current_time()

from resnet1d import resnet18, resnet34, resnet50, resnet101, resnet152

model_dict = {
    'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
    'resnet101': resnet101, 'resnet152': resnet152,
}


def train(model, device, train_loader, optimizer, scheduler, epoch, writer, log_interval, logger):
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct_cnt = 0
    total_cnt = 0
    
    for batch_idx, (data, target, ids) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_cnt += data.size(0)
        correct_cnt += (output.argmax(dim=1) == target).sum().item()
        
        if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_idx) 
    
    scheduler.step()
    logger.info('Train Accuracy: {:.2f}% ({}/{})'.format(100. * correct_cnt / total_cnt, correct_cnt, total_cnt))
    if writer is not None:
        writer.add_scalar('train/accuracy', 100. * correct_cnt / total_cnt, epoch)
    return correct_cnt / total_cnt


def test(model, device, test_loader, epoch, writer, logger, category_num):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    per_category_correct_cnt = np.zeros(category_num)
    per_category_total_cnt = np.zeros(category_num)
    with torch.no_grad():
        for data, target, ids in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_cnt += data.size(0)
            correct_cnt += (output.argmax(dim=1) == target).sum().item()
            # calculate per-class accuracy
            for i in range(category_num):
                per_category_total_cnt[i] += (target == i).sum().item()
                per_category_correct_cnt[i] += ((output.argmax(dim=1) == target) & (target == i)).sum().item()

    per_category_acc = per_category_correct_cnt / per_category_total_cnt
    logger.info('Per-Category Acc: {}'.format(per_category_acc))
    logger.info('Per-Category correct cnt: {}'.format(per_category_correct_cnt))
    acc = correct_cnt / total_cnt
    logger.info('Test Accuracy: {:.2f}% ({}/{})'.format(100. * correct_cnt / total_cnt, correct_cnt, total_cnt))
    writer.add_scalar('test/accuracy', acc, epoch)
    return acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # set up tensorboard writer
    checkpoint_dir = os.path.join(args.checkpoint_root_dir, f'ssim{args.ssim_threshold}-{args.model_name}-seed{args.seed}-{CURRENT_TIME}')
    writer = SummaryWriter(log_dir=checkpoint_dir)
    logger.info(f"Writing tensorboard logs to {checkpoint_dir}") 
    
    # set up model
    model = model_dict[args.model_name](args.num_labels, args.freq_bins, args.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    
    # dataset
    real_train_dataset = RealDFSDataset(args.real_data_dir, args.real_fname_label_dir, 'train', 
        args.train_rxs, args.normalize_dfs, return_ids=True)
    real_valid_dataset = RealDFSDataset(args.real_data_dir, args.real_fname_label_dir, 'valid', 
        args.test_rxs, args.normalize_dfs, return_ids=True)
    real_test_dataset = RealDFSDataset(args.real_data_dir, args.real_fname_label_dir, 'test', 
        args.test_rxs, args.normalize_dfs, return_ids=True)
    syn_train_dataset = SynDFSDataset(args.syn_data_dir, args.syn_fname_label_dir, 'train', 
        args.train_rxs, args.normalize_dfs, return_ids=True)
    valid_loader = DataLoader(real_valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(real_test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # filter syn_dataset with ssim to real data
    filter_indices = []
    for i in tqdm(range(len(real_train_dataset))):
        ori_array = real_train_dataset[i][0].numpy()
        syn_array = syn_train_dataset[i][0].numpy()
        ssim_index = real_ssim(ori_array, syn_array)
        if ssim_index > args.ssim_threshold:
            filter_indices.append(i) 
    selected_syn_dataset = SubsetDataset(syn_train_dataset, filter_indices)
    logger.info(f"Select {len(selected_syn_dataset)} synthetic data with TRTS")
    
    concat_dataset = ConcatDataset([real_train_dataset, selected_syn_dataset])
    train_loader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True)
    
    best_val_acc = 0.0
    best_model = None
    for epoch in range(0, args.epochs):
        train_acc = train(model, device, train_loader, optimizer, scheduler, epoch, writer, args.log_interval, logger)
        val_acc = test(model, device, valid_loader, epoch, writer, logger, args.num_labels)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            
    test_acc = test(best_model, device, test_loader, epoch, writer, logger, args.num_labels)
    torch.save(best_model.state_dict(), os.path.join(checkpoint_dir, f"test{test_acc:.4f}_val{best_val_acc:.4f}.pth"))
    writer.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--real_data_dir', type=str, default='../../0_real_data_preparation/real_dfs_data', help='Path to the directory containing the real data.')
    parser.add_argument('--real_fname_label_dir', type=str, default='../../0_real_data_preparation/real_fname_label', help='Path to the directory containing the label files.')
    parser.add_argument('--syn_data_dir', type=str, default='../../1_synthetic_data_generation/rf-diffusion/syn_data_native', help='Path to the directory containing the real data.')
    parser.add_argument('--syn_fname_label_dir', type=str, default='../../0_real_data_preparation/real_fname_label', help='Path to the directory containing the label files.')
    parser.add_argument('--train_rxs', default=[1,3,5], help='selected rx', 
        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--test_rxs', default=[2,4,6], help='selected rx', 
        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--normalize_dfs', action='store_true', help='Whether to normalize the DFS data.')
    parser.add_argument('--freq_bins', type=int, default=121, help='Number of frequency bins in the DFS.')
    parser.add_argument('--num_labels', type=int, default=6, help='Number of classes to classify.')
    
    # model parameters
    parser.add_argument('--model_name', type=str, default='resnet34', help='Name of the model to use.')
    # training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training.')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='step to do learning rate decay, 0 means no decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='learning rate decay rate')
    parser.add_argument('--seed', type=int, default=420, help='random seed to use')
    parser.add_argument('--ssim_threshold', type=float, default=0.5, help='Threshold for filtering synthetic data with SSIM.')
    
    # log parameters
    parser.add_argument('--log_dir', type=str, default='./rfdiffusion_filterssim_logs', help='Path to the directory to save logs.')
    parser.add_argument('--checkpoint_root_dir', type=str, default='./rfdiffusion_filterssim_checkpoints', help='Path to the directory to save checkpoints.')
    parser.add_argument('--log_interval', type=int, default=100, help='Interval for logging training status.')
    
    args = parser.parse_args()
    
    main(args)