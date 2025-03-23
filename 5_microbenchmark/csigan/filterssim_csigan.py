import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Dataset import SignFiCSIDataset
from gan_nn import Discriminator, Generator
from utils import get_current_time
from tqdm import tqdm
import argparse

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


# 3 datasets: real user2,3, real user1 unlabeled, synthetic user1 loader (treat it as labeled in original CsiGAN)
def train(args, labeled_dataset, unlabeled_real_dataset, syn_dataset, test_loader, 
          model, generator, optimizer_dis, optimizer_gen, scheduler_dis, scheduler_gen, 
          device, logger, writer):
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
    labeled_iter = iter(labeled_loader)
    unlabeled_real_loader = DataLoader(unlabeled_real_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    unlabeled_real_iter = iter(unlabeled_real_loader)
    syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, shuffle=True)
    syn_iter = iter(syn_loader)
    
    # statistics
    best_val_acc = 0
    
    # label for synthetic data and unlabeled real data
    genLabel = torch.cat(
        [torch.zeros(args.batch_size, args.category_num//2), 
        torch.ones(args.batch_size, args.category_num//2+1)], 
        dim=1).to(device)
    unlLabel = torch.cat(
        [torch.ones(args.batch_size, args.category_num//2), 
        torch.zeros(args.batch_size, args.category_num//2+1)], 
        dim=1).to(device) 
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        model.train()
        for batch_idx in range(args.eval_step):
            # data loading
            try:
                labeled_x, labeled_target = labeled_iter.__next__()
            except:
                labeled_iter = iter(labeled_loader)
                labeled_x, labeled_target = labeled_iter.__next__()
            try:
                unlabeled_real_x, _ = unlabeled_real_iter.__next__()
            except:
                unlabeled_real_iter = iter(unlabeled_real_loader)
                unlabeled_real_x, _ = unlabeled_real_iter.__next__()
            try:
                syn_x, syn_target = syn_iter.__next__()
            except:
                syn_iter = iter(syn_loader)
                syn_x, syn_target = syn_iter.__next__()
            labeled_x, labeled_target = labeled_x.to(device), labeled_target.to(device)
            unlabeled_real_x = unlabeled_real_x.to(device)
            syn_x, syn_target = syn_x.to(device), syn_target.to(device)
            
            # feed data to model
            # vanilla z generation
            random_z = torch.rand(args.batch_size, 100, device=device)
            vanilla_gen_x = generator(random_z)
            
            # train discriminator
            optimizer_dis.zero_grad()
            labeled_logits, _ = model(labeled_x)
            unlabeled_real_logits, _ = model(unlabeled_real_x)
            syn_logits, _ = model(syn_x)
            vanilla_gen_logits, _ = model(vanilla_gen_x.detach())
            # classify real labeled data
            labeled_loss = criterion(labeled_logits, labeled_target)
            # classify cycleGAN generated data
            syn_loss = criterion(syn_logits, syn_target)
            # classify unlabeled real data
            unlabeled_pred = F.softmax(unlabeled_real_logits, dim=1)
            vanilla_gen_pred = F.softmax(vanilla_gen_logits, dim=1)
            unlabeled_real_loss = -args.unlabeled_real_lambda * torch.mean(torch.mul(unlabeled_pred, unlLabel))
            vanilla_gen_loss = -args.vanilla_gen_lambda * torch.mean(torch.mul(vanilla_gen_pred, genLabel))
            # print("vanilla_gen_pred: ", vanilla_gen_pred[0], vanilla_gen_loss.item())
            # print("syn_pred: ", syn_logits[0], syn_target[0], syn_loss.item())
            unlabeled_loss = unlabeled_real_loss + args.cycle_lambda*syn_loss + vanilla_gen_loss
            
            # overall discriminator loss
            loss_dis = args.unlabeled_weight * unlabeled_loss + args.labeled_weight * labeled_loss
            loss_dis.backward()
            optimizer_dis.step()
            
            # train generator
            optimizer_gen.zero_grad()
            _, layer_fake = model(vanilla_gen_x)
            _, layer_real = model(unlabeled_real_x)
            m1 = torch.mean(layer_real, dim=0)
            m2 = torch.mean(layer_fake, dim=0)
            loss_gen = torch.mean(torch.abs(m1 - m2))
            loss_gen.backward()
            optimizer_gen.step()
            
            if batch_idx % args.log_interval == 0:
                writer.add_scalar('train/loss_dis', loss_dis.item(), epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/loss_gen', loss_gen.item(), epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/loss_labeled', labeled_loss.item(), epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/loss_unlabeled_real', unlabeled_real_loss.item(), epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/loss_cycle_syn', syn_loss.item(), epoch*args.eval_step+batch_idx)
                writer.add_scalar('train/loss_vanilla_gen', vanilla_gen_loss.item(), epoch*args.eval_step+batch_idx)
                logger_message = f"Train Epoch: {epoch} [{batch_idx*args.batch_size}/{args.eval_step*args.batch_size} ({100. * batch_idx / args.eval_step:.0f}%)]" + \
                    f"\tLoss_dis: {loss_dis.item():.6f}\tLoss_gen: {loss_gen.item():.6f}" + \
                    f"\tLoss_labeled: {labeled_loss.item():.6f}\tLoss_unlabeled_real: {unlabeled_real_loss.item():.6f}" + \
                    f"\tLoss_cycle_syn: {syn_loss.item():.6f}\tLoss_vanilla_gen: {vanilla_gen_loss.item():.6f}"
                logger.info(logger_message)
        
        # update learning rate
        scheduler_dis.step()
        scheduler_gen.step()
        
        val_acc = test(test_loader, model, epoch, device, logger, writer, args.category_num//2)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{args.checkpoint_dir}/epoch{epoch}_{val_acc:.4f}.pth')
    print(f"Best Val Acc: {best_val_acc:.4f}")


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
    
    ratio_name = f'syn{args.syn_ratio}'
    data_name = 'normcsi' if args.normalize_csi else 'rawcsi'
    ssim_name = f'ssim{args.ssim_threshold}'
    model_name = f'channel{args.mid_channels}'
    seed_name = f'seed{args.seed}'
    weight_name = f'l{args.labeled_weight}_u{args.unlabeled_weight}_unlabeledreal{args.unlabeled_real_lambda}_cycle{args.cycle_lambda}_vanillagen{args.vanilla_gen_lambda}'
    checkpoint_dir = os.path.join(args.checkpoint_root_dir, \
        f'{ratio_name}-{data_name}-{ssim_name}-{model_name}-{seed_name}-{weight_name}-{CURRENT_TIME}')
    writer = SummaryWriter(log_dir=checkpoint_dir)
    args.checkpoint_dir = checkpoint_dir
    logger.info(f"Writing tensorboard logs to {checkpoint_dir}")
    
    # set up model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Discriminator(args.seed, args.category_num, args.mid_channels)
    model.to(device)
    # vanilla generator
    generator = Generator().to(device)
    if args.model_ckpt_path is not None:
        model.load_state_dict(torch.load(args.model_ckpt_path, map_location=device))
    optimizer_dis = optim.Adam(model.parameters(), lr=args.lr)
    scheduler_dis = optim.lr_scheduler.StepLR(optimizer_dis, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    optimizer_gen = optim.Adam(generator.parameters(), lr=args.lr)
    scheduler_gen = optim.lr_scheduler.StepLR(optimizer_gen, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    
    # set up dataset
    labeled_dataset = SignFiCSIDataset(args.dataset_dir, \
        'train_csi', 'train_label', args.normalize_csi)
    unlabeled_real_dataset = SignFiCSIDataset(args.dataset_dir, \
        'train_leaveout_unlabeled_csi', 'train_leaveout_unlabeled_label', args.normalize_csi)
    test_dataset = SignFiCSIDataset(args.dataset_dir, \
        'leaveout_test_csi', 'leaveout_test_label', args.normalize_csi)
    if args.syn_ratio != 1.0:
        syn_dataset = SignFiCSIDataset(args.syn_dataset_dir, \
            'cycle_target_all_syn_csi', 'cycle_target_all_syn_label', args.normalize_csi, add_half=(args.category_num//2))
        selected_syn_cnt = int(args.syn_ratio * len(syn_dataset) / 5)
        np.random.seed(args.seed)
        selected_indices = np.random.choice(len(syn_dataset), selected_syn_cnt, replace=False)
        syn_dataset.set_index(selected_indices)
    else:
        syn_dataset = SignFiCSIDataset(args.ori_syn_dataset_dir, \
            'cycle_target_all_syn_csi', 'cycle_target_all_syn_label', args.normalize_csi, add_half=(args.category_num//2))
    
    # filter syn_dataset with ssim to real data
    syn_cnt = len(syn_dataset)
    real_cnt = len(unlabeled_real_dataset)
    ssim_scores = []
    for si in range(syn_cnt):
        syn_csi = syn_dataset[si][0].numpy()
        for ri in range(real_cnt):
            real_csi = unlabeled_real_dataset[ri][0].numpy()
            ssim_score = real_ssim(syn_csi, real_csi)
            if ri == 0:
                ssim_scores.append(ssim_score)
            else:
                ssim_scores[si] = max(ssim_scores[si], ssim_score)
    ssim_scores = np.array(ssim_scores)
    print(ssim_scores)
    filter_indices = np.where(ssim_scores > args.ssim_threshold)[0]
    if args.syn_ratio != 1.0:
        filter_indices = selected_indices[filter_indices]
    syn_dataset.set_index(filter_indices)
    logger.info(f"Select {len(syn_dataset)} synthetic data with ssim threshold {args.ssim_threshold}")

    test_loader = DataLoader(test_dataset, \
        batch_size=args.batch_size, shuffle=False)
    
    train(args, labeled_dataset, unlabeled_real_dataset, syn_dataset, test_loader, 
        model, generator, optimizer_dis, optimizer_gen, scheduler_dis, scheduler_gen, 
        device, logger, writer)
    
    # close tensorboard writer
    writer.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Reproduce CsiGAN')
    # data parameters
    parser.add_argument('--dataset_dir', type=str, default='../../0_real_data_preparation/csigan_data', help='path to original data')
    parser.add_argument('--ori_syn_dataset_dir', type=str, default='../../0_real_data_preparation/csigan_data', help='path to original data')
    parser.add_argument('--syn_dataset_dir', type=str, default='more_synthetic_data', help='path to original data')
    parser.add_argument('--normalize_csi', action='store_true', help='normalize csi')
    parser.add_argument('--category_num', type=int, default=101, help='number of classes')
    parser.add_argument('--syn_ratio', type=float, default=1.0, help='ratio of synthetic data to real data')
    # model parameters
    parser.add_argument('--mid_channels', type=int, default=128, help='number of channels in the intermediate layers')
    parser.add_argument('--model_ckpt_path', type=str, default=None, help='path to pre-trained model')
    # training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--mu', default=1, type=int, help='number of unlabeled samples per labeled sample')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_decay_step', type=int, default=40, help='step to do learning rate decay, 0 means no decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--eval_step', type=int, default=100, help='evaluation step')
    parser.add_argument('--seed', type=int, default=420, help='random seed to use')
    # ssim filtering parameters
    parser.add_argument('--ssim_threshold', type=float, default=0.5, help='ssim threshold for filtering synthetic data')
    
    # log parameters
    parser.add_argument('--log_dir', type=str, default='./csigan_filterssim_logs', help='path to log directory')
    parser.add_argument('--checkpoint_root_dir', type=str, default='./csigan_filterssim_checkpoints', help='path to checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    
    # loss parameters
    parser.add_argument('--labeled_weight', type=float, default=1.0, help='labeled data weight')
    parser.add_argument('--unlabeled_weight', type=float, default=1.0, help='unlabeled data weight')
    parser.add_argument('--unlabeled_real_lambda', type=float, default=0.6, help='lambda for unlabeled real data')
    parser.add_argument('--cycle_lambda', type=float, default=0.1, help='lambda for cycle consistency loss')
    parser.add_argument('--vanilla_gen_lambda', type=float, default=0.1, help='lambda for vanilla generator loss')
    args = parser.parse_args()
    main(args)
