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

CURRENT_TIME: str = get_current_time()

# 3 datasets: real user2,3, real user1 unlabeled
def train(args, labeled_dataset, unlabeled_real_dataset, test_loader, 
          model, generator, optimizer_dis, optimizer_gen, scheduler_dis, scheduler_gen, 
          device, logger, writer):
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
    labeled_iter = iter(labeled_loader)
    unlabeled_real_loader = DataLoader(unlabeled_real_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    unlabeled_real_iter = iter(unlabeled_real_loader)
    
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
            
            labeled_x, labeled_target = labeled_x.to(device), labeled_target.to(device)
            unlabeled_real_x = unlabeled_real_x.to(device)
            
            # feed data to model
            # vanilla z generation
            random_z = torch.rand(args.batch_size, 100, device=device)
            vanilla_gen_x = generator(random_z)
            
            # train discriminator
            optimizer_dis.zero_grad()
            labeled_logits, _ = model(labeled_x)
            unlabeled_real_logits, _ = model(unlabeled_real_x)
            vanilla_gen_logits, _ = model(vanilla_gen_x.detach())
            # classify real labeled data
            labeled_loss = criterion(labeled_logits, labeled_target)
            # classify unlabeled real data
            unlabeled_pred = F.softmax(unlabeled_real_logits, dim=1)
            vanilla_gen_pred = F.softmax(vanilla_gen_logits, dim=1)
            unlabeled_real_loss = -args.unlabeled_real_lambda * torch.mean(torch.mul(unlabeled_pred, unlLabel))
            vanilla_gen_loss = -args.vanilla_gen_lambda * torch.mean(torch.mul(vanilla_gen_pred, genLabel))
            # print("vanilla_gen_pred: ", vanilla_gen_pred[0], vanilla_gen_loss.item())
            # print("syn_pred: ", syn_logits[0], syn_target[0], syn_loss.item())
            unlabeled_loss = unlabeled_real_loss + vanilla_gen_loss
            
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
                writer.add_scalar('train/loss_vanilla_gen', vanilla_gen_loss.item(), epoch*args.eval_step+batch_idx)
                logger_message = f"Train Epoch: {epoch} [{batch_idx*args.batch_size}/{args.eval_step*args.batch_size} ({100. * batch_idx / args.eval_step:.0f}%)]" + \
                    f"\tLoss_dis: {loss_dis.item():.6f}\tLoss_gen: {loss_gen.item():.6f}" + \
                    f"\tLoss_labeled: {labeled_loss.item():.6f}\tLoss_unlabeled_real: {unlabeled_real_loss.item():.6f}" + \
                    f"\tLoss_vanilla_gen: {vanilla_gen_loss.item():.6f}"
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
    
    data_name = 'normcsi' if args.normalize_csi else 'rawcsi'
    model_name = f'channel{args.mid_channels}'
    seed_name = f'seed{args.seed}'
    weight_name = f'l{args.labeled_weight}_u{args.unlabeled_weight}_unlabeledreal{args.unlabeled_real_lambda}_vanillagen{args.vanilla_gen_lambda}'
    checkpoint_dir = os.path.join(args.checkpoint_root_dir, \
        f'{data_name}-{model_name}-{seed_name}-{weight_name}-{CURRENT_TIME}')
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
    test_loader = DataLoader(test_dataset, \
        batch_size=args.batch_size, shuffle=False)
    
    train(args, labeled_dataset, unlabeled_real_dataset, test_loader, 
        model, generator, optimizer_dis, optimizer_gen, scheduler_dis, scheduler_gen, 
        device, logger, writer)
    
    # close tensorboard writer
    writer.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Reproduce CsiGAN')
    parser.add_argument('--dataset_dir', type=str, default='../../0_real_data_preparation/csigan_data', help='path to original data')
    parser.add_argument('--normalize_csi', action='store_true', help='normalize csi')
    parser.add_argument('--category_num', type=int, default=101, help='number of classes')
    # model parameters
    parser.add_argument('--mid_channels', type=int, default=128, help='number of channels in the intermediate layers')
    parser.add_argument('--model_ckpt_path', type=str, default=None, help='path to pre-trained model')
    # training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--mu', default=1, type=int, help='number of unlabeled samples per labeled sample')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--lr_decay_step', type=int, default=40, help='step to do learning rate decay, 0 means no decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--eval_step', type=int, default=100, help='evaluation step')
    parser.add_argument('--seed', type=int, default=420, help='random seed to use')
    
    # log parameters
    parser.add_argument('--log_dir', type=str, default='./csigan_onlyreal_logs', help='path to log directory')
    parser.add_argument('--checkpoint_root_dir', type=str, default='./csigan_onlyreal_checkpoints', help='path to checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    
    # loss parameters
    parser.add_argument('--labeled_weight', type=float, default=1.0, help='labeled data weight')
    parser.add_argument('--unlabeled_weight', type=float, default=1.0, help='unlabeled data weight')
    parser.add_argument('--unlabeled_real_lambda', type=float, default=0.6, help='lambda for unlabeled real data')
    parser.add_argument('--vanilla_gen_lambda', type=float, default=0.1, help='lambda for vanilla generator loss')
    args = parser.parse_args()
    main(args)
    
