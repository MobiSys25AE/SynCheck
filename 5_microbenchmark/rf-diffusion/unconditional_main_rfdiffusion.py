import logging
import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_transform import GaussianTransform
from Dataset import RealDFSDataset, SynDFSDataset
from resnet1d_syncheck import resnet18, resnet34, resnet50, resnet101, resnet152
from losses import ova_loss, ova_ent
from data_selection import exclude_dataset
from utils import AverageMeter, get_current_time
from tqdm import tqdm
import argparse

CURRENT_TIME: str = get_current_time()

model_dict = {
    'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
    'resnet101': resnet101, 'resnet152': resnet152,
}


def train(args, labeled_dataset, unlabeled_dataset, test_loader, val_loader, 
          model, optimizer, scheduler, device, logger, writer):
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
    labeled_iter = iter(labeled_loader)
    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
    
    # statistics
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_o = AverageMeter()
    losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs = AverageMeter()
    best_val_acc = 0
    best_model = None
    
    for epoch in range(args.epochs):
        if epoch >= args.start_fix:
            tmp_batch_size = args.batch_size
            args.batch_size = 2 * (args.mu + 1) * tmp_batch_size
            exclude_dataset(args, unlabeled_dataset, model, device, logger=logger)
            args.batch_size = tmp_batch_size
        
        model.train()
        # initialize train loaders to unlabeled dataset
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size*args.mu, shuffle=True)
        unlabeled_loader_all = DataLoader(unlabeled_dataset_all, batch_size=args.batch_size*args.mu, shuffle=True)
        unlabeled_iter = iter(unlabeled_loader)
        unlabeled_iter_all = iter(unlabeled_loader_all)
        
        for batch_idx in range(args.eval_step):
            # data loading
            try:
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.__next__()
            except:
                labeled_iter = iter(labeled_loader)
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.__next__()
            try:
                (inputs_u_w, inputs_u_s, _), _ = unlabeled_iter.__next__()
            except:
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_u_w, inputs_u_s, _), _ = unlabeled_iter.__next__()
            try:
                (inputs_u_w_all, inputs_u_s_all, _), _ = unlabeled_iter_all.__next__()
            except:
                unlabeled_iter_all = iter(unlabeled_loader_all)
                (inputs_u_w_all, inputs_u_s_all, _), _ = unlabeled_iter_all.__next__()
                
            cur_bs = inputs_x.size(0)
            inputs_unlabeled_all = torch.cat([inputs_u_w_all, inputs_u_s_all], dim=0)
            inputs = torch.cat([inputs_x_s, inputs_x, inputs_unlabeled_all], dim=0).to(device)
            targets_x = targets_x.to(device)
            # feed data
            logits, logits_open = model(inputs)
            logits_open_u1, logits_open_u2 = logits_open[2*cur_bs:].chunk(2)
            
            # labeled loss
            Lx = F.cross_entropy(logits[:2*cur_bs], targets_x.repeat(2), reduction='mean')
            Lo = ova_loss(logits_open[:2*cur_bs], targets_x.repeat(2))
            
            # open-set entropy minimization
            L_oem = ova_ent(logits_open_u1) / 2.
            L_oem += ova_ent(logits_open_u2) / 2.
            
            # soft consistency regularization
            logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
            logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
            logits_open_u1 = F.softmax(logits_open_u1, 1)
            logits_open_u2 = F.softmax(logits_open_u2, 1)
            L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
                logits_open_u1 - logits_open_u2)**2, 1), 1))
            
            # pseudo label
            if epoch >= args.start_fix:
                inputs_u_ws = torch.cat([inputs_u_w, inputs_u_s], dim=0).to(device)
                inputs_u_ws_repeat = torch.cat([inputs_x_s.to(device), inputs_x.to(device), inputs_u_ws], dim=0).to(device)
                logits, _ = model(inputs_u_ws_repeat)
                logits = logits[2*cur_bs:]
                logits_u_w, logits_u_s = logits.chunk(2)
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                L_fix = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
                mask_probs.update(mask.mean().item())
            else:
                L_fix = torch.zeros(1).to(device).mean()
            
            # total loss
            optimizer.zero_grad()
            loss = Lx + args.lambda_ova * Lo + args.lambda_oem * L_oem \
                + args.lambda_socr * L_socr + args.lambda_fix * L_fix
            loss.backward()
            optimizer.step()
            
            # update statistics
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_o.update(Lo.item())
            losses_oem.update(L_oem.item())
            losses_socr.update(L_socr.item())
            losses_fix.update(L_fix.item())
            
            if batch_idx % args.log_interval == 0:
                logger_message = f'Epoch: {epoch+1}/{args.epochs}\t' + f'Batch: {batch_idx+1}/{args.eval_step}\t' \
                    + f'train loss {losses.avg:.4f}\t' + f'Lx {losses_x.avg:.4f}\t' \
                    + f'Lo {losses_o.avg:.4f}\t' + f'L_oem {losses_oem.avg:.4f}\t' \
                    + f'L_socr {losses_socr.avg:.4f}\t' + f'L_fix {losses_fix.avg:.4f}\t' \
                    + f'Mask_probs {mask_probs.avg:.4f}\t'
                logger.info(logger_message)
                writer.add_scalar('train/loss', losses.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/Lx', losses_x.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/Lo', losses_o.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/L_oem', losses_oem.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/L_socr', losses_socr.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/L_fix', losses_fix.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/mask_probs', mask_probs.avg, epoch*args.eval_step+batch_idx)
        
        # update learning rate
        scheduler.step()
        
        # validation
        val_acc = test(val_loader, model, epoch, device, logger, writer)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
        
    # test
    test_acc = test(test_loader, best_model, epoch, device, logger, writer)
    torch.save(best_model.state_dict(), os.path.join(args.checkpoint_dir, f'test{test_acc:.4f}_val{best_val_acc:.4f}.pth'))


def test(val_loader, model, epoch, device, logger, writer):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            total_cnt += targets.size(0)
            correct_cnt += predicted.eq(targets).sum().item()
    acc = correct_cnt / total_cnt
    logger.info(f'Epoch: {epoch+1} Test Acc {acc:.4f}\t')
    writer.add_scalar('test/acc', acc, epoch)
    return acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up logger and tensorboard writer
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
    
    seed_name = f'seed{args.seed}'
    data_name = 'normdfs' if args.normalize_dfs else 'rawdfs'
    threshold_name = f'thresh{args.threshold}'
    temp_name = f'temp{args.T}'
    weight_name = f'ova{args.lambda_ova}_oem{args.lambda_oem}_socr{args.lambda_socr}_fix{args.lambda_fix}'
    checkpoint_dir = os.path.join(args.checkpoint_root_dir, \
        f'unconditional-{seed_name}-{data_name}-{args.model_name}-{threshold_name}-{temp_name}-{weight_name}-{CURRENT_TIME}')
    writer = SummaryWriter(log_dir=checkpoint_dir)
    args.checkpoint_dir = checkpoint_dir
    logger.info(f"Writing tensorboard logs to {checkpoint_dir}")
    
    # set up model
    model = model_dict[args.model_name](args.num_labels, args.freq_bins, args.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    
    # set up datasets
    real_transform = GaussianTransform(args.weak_mean, args.weak_std, args.strong_mean, args.strong_std, args.seed)
    syn_transform = GaussianTransform(args.weak_mean, args.weak_std, args.strong_mean, args.strong_std, args.seed)
    labeled_traindataset = RealDFSDataset(args.real_data_dir, args.real_fname_label_dir, 'train', \
        args.train_rxs, args.normalize_dfs, real_transform)
    labeled_valdataset = RealDFSDataset(args.real_data_dir, args.real_fname_label_dir, 'valid', \
        args.test_rxs, args.normalize_dfs)
    labeled_testdataset = RealDFSDataset(args.real_data_dir, args.real_fname_label_dir, 'test', \
        args.test_rxs, args.normalize_dfs)
    unlabeled_dataset = SynDFSDataset(args.syn_data_dir, args.syn_fname_label_dir, 'train', \
        args.train_rxs, args.normalize_dfs, syn_transform)
    val_loader = DataLoader(labeled_valdataset, batch_size=args.batch_size*2*(args.mu+1), shuffle=False)
    test_loader = DataLoader(labeled_testdataset, batch_size=args.batch_size*2*(args.mu+1), shuffle=False)
    
    # train
    train(args, labeled_traindataset, unlabeled_dataset, test_loader, val_loader, 
          model, optimizer, scheduler, device, logger, writer)
    
    # close tensorboard writer
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--real_data_dir', type=str, default='../../0_real_data_preparation/real_dfs_data', help='Path to the directory containing the real data.')
    parser.add_argument('--real_fname_label_dir', type=str, default='../../0_real_data_preparation/real_fname_label', help='Path to the directory containing the label files.')
    parser.add_argument('--syn_data_dir', type=str, default='uncond_syn_data_lownoise_native', help='Path to the directory containing the synthetic data.')
    parser.add_argument('--syn_fname_label_dir', type=str, default='../../0_real_data_preparation/real_fname_label', help='Path to the directory containing the label files.')
    parser.add_argument('--train_rxs', default=[1,3,5], help='selected rx', 
        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--test_rxs', default=[2,4,6], help='selected rx', 
        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--normalize_dfs', action='store_true', help='normalize dfs')
    parser.add_argument('--weak_mean', type=float, default=0.0, help='weak augmentation mean')
    parser.add_argument('--weak_std', type=float, default=0.01, help='weak augmentation std')
    parser.add_argument('--strong_mean', type=float, default=0.0, help='strong augmentation mean')  
    parser.add_argument('--strong_std', type=float, default=0.03, help='strong augmentation std')
    parser.add_argument('--num_labels', type=int, default=6, help='Number of classes to classify.')
    # model parameters
    parser.add_argument('--freq_bins', type=int, default=121, help='number of frequency bins in dfs')
    parser.add_argument('--model_name', type=str, default='resnet34', help='name of 1d model')
    # training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--mu', default=1, type=int, help='number of unlabeled samples per labeled sample')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training.')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='step to do learning rate decay, 0 means no decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='learning rate decay rate')
    parser.add_argument('--lambda_ova', type=float, default=1.0, help='lambda for ova')
    parser.add_argument('--lambda_oem', type=float, default=0.1, help='lambda for open-set entropy minimization')
    parser.add_argument('--lambda_socr', type=float, default=0.5, help='lambda for soft consistency regularization')
    parser.add_argument('--lambda_fix', type=float, default=1.0, help='lambda for pseudo label')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for pseudo label')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for pseudo label')
    parser.add_argument('--eval_step', type=int, default=420, help='evaluation step')
    parser.add_argument('--start_fix', type=int, default=5, help='start fix epoch')
    parser.add_argument('--seed', type=int, default=420, help='random seed to use')
    # log parameters
    parser.add_argument('--log_dir', type=str, default='./unconditional_rfdiffusion_syncheck_logs', help='path to log directory')
    parser.add_argument('--checkpoint_root_dir', type=str, default='./unconditional_rfdiffusion_syncheck_checkpoints', help='path to checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    args = parser.parse_args()
    main(args)