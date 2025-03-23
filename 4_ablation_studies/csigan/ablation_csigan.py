import logging
import os
import sys
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_transform import GaussianTransform
from Dataset import SignFiCSIDataset
from gan_nn import Discriminator_syncheck
from losses import ova_loss, ova_ent
from data_selection import exclude_dataset
from utils import AverageMeter, get_current_time
from tqdm import tqdm
import argparse

CURRENT_TIME: str = get_current_time()

# 3 datasets: real user2,3, real user1 unlabeled, synthetic user1
def train(args, labeled_dataset, unlabeled_real_dataset, unlabeled_syn_dataset, test_loader, 
          model, optimizer, scheduler, device, logger, writer):
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
    labeled_iter = iter(labeled_loader)
    unlabeled_real_dataset_all = copy.deepcopy(unlabeled_real_dataset)
    unlabeled_syn_dataset_all = copy.deepcopy(unlabeled_syn_dataset)
    
    # statistics
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_o = AverageMeter()
    losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs_real = AverageMeter()
    mask_probs_syn = AverageMeter()
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        if epoch >= args.start_fix:
            exclude_dataset(args, unlabeled_real_dataset, model, device, logger=logger)
            exclude_dataset(args, unlabeled_syn_dataset, model, device, logger=logger)
        
        model.train()
        # initialize train loaders to unlabeled datasets
        unlabeled_real_loader = DataLoader(unlabeled_real_dataset, batch_size=args.batch_size*args.mu, shuffle=True)
        unlabeled_real_loader_all = DataLoader(unlabeled_real_dataset_all, batch_size=args.batch_size*args.mu, shuffle=True)
        unlabeled_syn_loader = DataLoader(unlabeled_syn_dataset, batch_size=args.batch_size*args.mu, shuffle=True)
        unlabeled_syn_loader_all = DataLoader(unlabeled_syn_dataset_all, batch_size=args.batch_size*args.mu, shuffle=True)
        
        unlabeled_real_iter = iter(unlabeled_real_loader)
        unlabeled_real_iter_all = iter(unlabeled_real_loader_all)
        unlabeled_syn_iter = iter(unlabeled_syn_loader)
        unlabeled_syn_iter_all = iter(unlabeled_syn_loader_all)
        
        for batch_idx in range(args.eval_step):
            # data loading
            try:
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.__next__()
            except:
                labeled_iter = iter(labeled_loader)
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.__next__()
            try:
                (inputs_u_real_w, inputs_u_real_s, _), targets_unlabeled_real = unlabeled_real_iter.__next__()
            except:
                unlabeled_real_iter = iter(unlabeled_real_loader)
                (inputs_u_real_w, inputs_u_real_s, _), targets_unlabeled_real = unlabeled_real_iter.__next__()
            try:
                (inputs_u_real_w_all, inputs_u_real_s_all, _), _ = unlabeled_real_iter_all.__next__()
            except:
                unlabeled_real_iter_all = iter(unlabeled_real_loader_all)
                (inputs_u_real_w_all, inputs_u_real_s_all, _), _ = unlabeled_real_iter_all.__next__()
            try:
                (inputs_u_syn_w, inputs_u_syn_s, _), targets_unlabeled_syn = unlabeled_syn_iter.__next__()
            except:
                unlabeled_syn_iter = iter(unlabeled_syn_loader)
                (inputs_u_syn_w, inputs_u_syn_s, _), targets_unlabeled_syn = unlabeled_syn_iter.__next__()
            try:
                (inputs_u_syn_w_all, inputs_u_syn_s_all, _), _ = unlabeled_syn_iter_all.__next__()
            except:
                unlabeled_syn_iter_all = iter(unlabeled_syn_loader_all)
                (inputs_u_syn_w_all, inputs_u_syn_s_all, _), _ = unlabeled_syn_iter_all.__next__()
            
            cur_bs = inputs_x.shape[0]
            cur_bs_real = inputs_u_real_w_all.shape[0]
            cur_bs_syn = inputs_u_syn_w_all.shape[0]
            inputs_unlabeled_real_all = torch.cat([inputs_u_real_w_all, inputs_u_real_s_all], dim=0)
            inputs_unlabeled_syn_all = torch.cat([inputs_u_syn_w_all, inputs_u_syn_s_all], dim=0)
            inputs = torch.cat([inputs_x_s, inputs_x, inputs_unlabeled_real_all, inputs_unlabeled_syn_all], dim=0).to(device)
            targets_x = targets_x.to(device)
            
            # feed data
            logits, logits_open = model(inputs)
            logits_open_u1_real, logits_open_u2_real = logits_open[2*cur_bs:2*cur_bs+2*cur_bs_real].chunk(2)
            logits_open_u1_syn, logits_open_u2_syn = logits_open[2*cur_bs+2*cur_bs_real:].chunk(2)
            
            # labeled loss
            Lx = F.cross_entropy(logits[:2*cur_bs], targets_x.repeat(2), reduction='mean')
            Lo = ova_loss(logits_open[:2*cur_bs], targets_x.repeat(2))
            
            # open-set entropy minimization
            L_oem = ova_ent(logits_open_u1_real) / 2.
            L_oem += ova_ent(logits_open_u2_real) / 2.
            L_oem += ova_ent(logits_open_u1_syn) / 2.
            L_oem += ova_ent(logits_open_u2_syn) / 2.
            
            # soft consistency regularization
            logits_open_u1_real = logits_open_u1_real.view(cur_bs_real, 2, -1)
            logits_open_u2_real = logits_open_u2_real.view(cur_bs_real, 2, -1)
            logits_open_u1_syn = logits_open_u1_syn.view(cur_bs_syn, 2, -1)
            logits_open_u2_syn = logits_open_u2_syn.view(cur_bs_syn, 2, -1)
            logits_open_u1_real = F.softmax(logits_open_u1_real, dim=1)
            logits_open_u2_real = F.softmax(logits_open_u2_real, dim=1)
            logits_open_u1_syn = F.softmax(logits_open_u1_syn, dim=1)
            logits_open_u2_syn = F.softmax(logits_open_u2_syn, dim=1)
            L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
                logits_open_u1_real - logits_open_u2_real)**2, 1), 1))
            L_socr += torch.mean(torch.sum(torch.sum(torch.abs(
                logits_open_u1_syn - logits_open_u2_syn)**2, 1), 1))
            
            # pseudo labels
            if epoch >= args.start_fix:
                cur_bs_real = inputs_u_real_w.shape[0]
                cur_bs_syn = inputs_u_syn_w.shape[0]
                inputs_u_real_ws = torch.cat([inputs_u_real_w, inputs_u_real_s], dim=0).to(device)
                inputs_u_syn_ws = torch.cat([inputs_u_syn_w, inputs_u_syn_s], dim=0).to(device)
                inputs_u_ws_repeat = torch.cat([inputs_x_s.to(device), inputs_x.to(device), \
                    inputs_u_real_ws, inputs_u_syn_ws], dim=0).to(device)
                logits, _ = model(inputs_u_ws_repeat)
                logits_u_w_real, logits_u_s_real = logits[2*cur_bs:2*cur_bs+2*cur_bs_real].chunk(2)
                logits_u_w_syn, logits_u_s_syn = logits[2*cur_bs+2*cur_bs_real:].chunk(2)
                pseudo_label_real = torch.softmax(logits_u_w_real.detach()/args.T, dim=-1)
                pseudo_label_syn = torch.softmax(logits_u_w_syn.detach()/args.T, dim=-1)
                max_probs_real, targets_u_real = torch.max(pseudo_label_real, dim=-1)
                max_probs_syn, targets_u_syn = torch.max(pseudo_label_syn, dim=-1)
                mask_real = max_probs_real.ge(args.threshold).float()
                mask_syn = max_probs_syn.ge(args.threshold).float()
                # targets_unlabeled_real = targets_unlabeled_real.to(device)
                # targets_unlabeled_syn = targets_unlabeled_syn.to(device)
                # L_fix = (F.cross_entropy(logits_u_s_real, targets_unlabeled_real, reduction='none')).mean()
                # L_fix += (F.cross_entropy(logits_u_s_syn, targets_unlabeled_syn, reduction='none')).mean()
                L_fix = (F.cross_entropy(logits_u_s_real, targets_u_real, reduction='none') * mask_real).mean()
                L_fix += (F.cross_entropy(logits_u_s_syn, targets_u_syn, reduction='none') * mask_syn).mean()
                mask_probs_real.update(mask_real.mean().item())
                mask_probs_syn.update(mask_syn.mean().item())
            else:
                L_fix = torch.zeros(1).to(device).mean()
            
            # total loss
            optimizer.zero_grad()
            loss = args.lambda_x*Lx + args.lambda_ova*Lo + args.lambda_oem * L_oem \
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
                # logger.info(logits[:2*cur_bs].argmax(dim=1))
                # logger.info(targets_x)
                logger_message = f'Epoch: {epoch+1}/{args.epochs}\t' + f'Batch: {batch_idx+1}/{args.eval_step}\t' \
                    + f'train loss {losses.avg:.4f}\t' + f'Lx {losses_x.avg:.4f}\t' \
                    + f'Lo {losses_o.avg:.4f}\t' + f'L_oem {losses_oem.avg:.4f}\t' \
                    + f'L_socr {losses_socr.avg:.4f}\t' + f'L_fix {losses_fix.avg:.4f}\t' \
                    + f'Mask_probs_real {mask_probs_real.avg:.4f}\t' \
                    + f'Mask_probs_syn {mask_probs_syn.avg:.4f}'
                # logger.info(logger_message)
                writer.add_scalar('train/loss', losses.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/Lx', losses_x.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/Lo', losses_o.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/L_oem', losses_oem.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/L_socr', losses_socr.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/L_fix', losses_fix.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/mask_probs_real', mask_probs_real.avg, epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/mask_probs_syn', mask_probs_syn.avg, epoch*args.eval_step+batch_idx)
        
        # update learning rate
        scheduler.step()
        
        val_acc = test(test_loader, model, epoch, device, logger, writer, args.num_classes)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{args.checkpoint_dir}/epoch{epoch}_{val_acc:.4f}.pth')


def test(val_loader, model, epoch, device, logger, writer, category_num):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    per_category_correct_cnt = np.zeros(category_num)
    per_category_total_cnt = np.zeros(category_num)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            total_cnt += targets.size(0)
            correct_cnt += predicted.eq(targets).sum().item()
            # calculate per-class accuracy
            for i in range(category_num):
                per_category_total_cnt[i] += (targets == i).sum().item()
                per_category_correct_cnt[i] += ((outputs.argmax(dim=1) == targets) & (targets == i)).sum().item()
    
    per_category_acc = per_category_correct_cnt / per_category_total_cnt
    logger.info('Per-Category Acc: {}'.format(per_category_acc))            
    acc = correct_cnt / total_cnt
    logger.info(f'Epoch: {epoch+1} Test Acc {acc:.4f}\t')
    writer.add_scalar('test/acc', acc, epoch)
    return acc


def main(args):
    torch.manual_seed(args.seed)
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

    seed_name = f'seed{args.seed}'
    data_name = 'normcsi' if args.normalize_csi else 'rawcsi'
    model_name = f'channel{args.mid_channels}'
    threshold_name = f'thresh{args.threshold}'
    temp_name = f'temp{args.T}'
    ablation_name = f'ova{args.lambda_ova}_oem{args.lambda_oem}_socr{args.lambda_socr}_fix{args.lambda_fix}'
    checkpoint_dir = os.path.join(args.checkpoint_root_dir, \
        f'{ablation_name}-{seed_name}-{data_name}-{model_name}-{threshold_name}-{temp_name}-{CURRENT_TIME}')
    writer = SummaryWriter(log_dir=checkpoint_dir)
    args.checkpoint_dir = checkpoint_dir
    logger.info(f"Writing tensorboard logs to {checkpoint_dir}")
    
    # set up model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Discriminator_syncheck(category=args.num_classes, mid_channels=args.mid_channels, seed=args.seed)
    model.to(device)
    if args.model_ckpt_path is not None:
        model.load_state_dict(torch.load(args.model_ckpt_path, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    
    # set up datasets
    labeled_transform = GaussianTransform(args.weak_mean, args.weak_std, args.strong_mean, args.strong_std, args.seed)
    real_transform = GaussianTransform(args.weak_mean, args.weak_std, args.strong_mean, args.strong_std, args.seed)
    syn_transform = GaussianTransform(args.weak_mean, args.weak_std, args.strong_mean, args.strong_std, args.seed)
    labeled_traindataset = SignFiCSIDataset(args.dataset_dir, \
        'train_csi', 'train_label', args.normalize_csi, labeled_transform)
    unlabeled_real_dataset = SignFiCSIDataset(args.dataset_dir, \
        'train_leaveout_unlabeled_csi', 'train_leaveout_unlabeled_label', args.normalize_csi, real_transform)
    unlabeled_syn_dataset = SignFiCSIDataset(args.dataset_dir, \
        'cycle_target_all_syn_csi', 'cycle_target_all_syn_label', args.normalize_csi, syn_transform)  # [TODO] whether to norm synthetic data
    test_dataset = SignFiCSIDataset(args.dataset_dir, \
        'leaveout_test_csi', 'leaveout_test_label', args.normalize_csi)
    test_loader = DataLoader(test_dataset, \
        batch_size=args.batch_size, shuffle=False)
    
    # train
    train(args, labeled_traindataset, unlabeled_real_dataset, unlabeled_syn_dataset, \
        test_loader, model, optimizer, scheduler, device, logger, writer)
    
    # close tensorboard writer
    writer.close()            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--dataset_dir', type=str, default='../../0_real_data_preparation/csigan_data', help='path to original data')
    parser.add_argument('--normalize_csi', action='store_true', help='normalize csi')
    parser.add_argument('--weak_mean', type=float, default=0.0, help='weak augmentation mean')
    parser.add_argument('--weak_std', type=float, default=0.0, help='weak augmentation std')
    parser.add_argument('--strong_mean', type=float, default=0.0, help='strong augmentation mean')  
    parser.add_argument('--strong_std', type=float, default=0.0001, help='strong augmentation std')
    parser.add_argument('--num_classes', type=int, default=50, help='number of classes')
    # model parameters
    parser.add_argument('--mid_channels', type=int, default=128, help='number of channels in the intermediate layers')
    parser.add_argument('--model_ckpt_path', type=str, default=None, help='path to pre-trained model')
    # training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--mu', default=1, type=int, help='number of unlabeled samples per labeled sample')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr_decay_step', type=int, default=40, help='step to do learning rate decay, 0 means no decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lambda_x', type=float, default=10.0, help='lambda for labeled loss')
    parser.add_argument('--lambda_ova', type=float, default=1.0, help='lambda for ova')
    parser.add_argument('--lambda_oem', type=float, default=0.1, help='lambda for open-set entropy minimization')
    parser.add_argument('--lambda_socr', type=float, default=0.5, help='lambda for soft consistency regularization')
    parser.add_argument('--lambda_fix', type=float, default=1.0, help='lambda for pseudo label')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for pseudo label')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for pseudo label')
    parser.add_argument('--eval_step', type=int, default=100, help='evaluation step')
    parser.add_argument('--start_fix', type=int, default=30, help='start fix epoch')
    parser.add_argument('--seed', type=int, default=420, help='random seed to use')
    
    # log parameters
    parser.add_argument('--log_dir', type=str, default='./csigan_ablation_logs', help='path to log directory')
    parser.add_argument('--checkpoint_root_dir', type=str, default='./csigan_ablation_checkpoints', help='path to checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    args = parser.parse_args()
    main(args)
    