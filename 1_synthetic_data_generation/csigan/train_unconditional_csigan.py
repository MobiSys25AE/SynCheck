import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
from cyclegan_model import CycleGANModel
from Dataset import CSIDataset, CycleDataset
from cyclegan_module import ImagePool
from utils import get_current_time


CURRENT_TIME: str = get_current_time()

def main(args):
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rng_data = np.random.RandomState(rng.randint(0, 2**10))
    
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
    
    # set up tensorboard writer
    scale_name = 'NormedCsi' if args.normalize_csi else 'RawCsi'
    checkpoint_dir = os.path.join(args.checkpoint_root_dir, 
        f'ShuffledCycleGAN-{scale_name}-{CURRENT_TIME}')
    writer = SummaryWriter(log_dir=checkpoint_dir)
    logger.info(f"Writing tensorboard logs to {checkpoint_dir}")
    
    source_dataset = CSIDataset(args.dataset_dir, \
        'cycle_source_csi', 'cycle_source_label', args.normalize_csi)
    target_dataset = CSIDataset(args.dataset_dir, \
        'cycle_target_csi', 'cycle_target_label', args.normalize_csi)
    source_all_dataset = CSIDataset(args.dataset_dir, \
        'cycle_source_all_csi', 'cycle_source_all_label', args.normalize_csi)
    target_all_dataset = CSIDataset(args.dataset_dir, \
        'cycle_target_all_csi', 'cycle_target_all_label', args.normalize_csi)
    cycle_dataset = CycleDataset(source_dataset, target_dataset, shuffle_pair=True)
    cycle_all_dataset = CycleDataset(source_all_dataset, target_all_dataset)
    cycle_loader = DataLoader(cycle_dataset, batch_size=args.batch_size, shuffle=True)
    cycle_all_loader = DataLoader(cycle_all_dataset, batch_size=args.batch_size, shuffle=False)
    
    # set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cycleGAN model
    model = CycleGANModel(args, device)
    image_pool = ImagePool(args.max_size)
    
    # loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    
    # optimizer
    optimizer_G = optim.Adam(list(model.G_A2B.parameters()) + list(model.G_B2A.parameters()), \
        lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D_A = optim.Adam(model.D_A.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D_B = optim.Adam(model.D_B.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    scheduler_D_A = optim.lr_scheduler.StepLR(optimizer_D_A, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    scheduler_D_B = optim.lr_scheduler.StepLR(optimizer_D_B, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    
    # training
    min_trained_l1 = np.inf
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch_data in enumerate(cycle_loader):
            source_csi, source_label, target_csi, target_label = batch_data
            source_csi, source_label, target_csi, target_label = \
                source_csi.to(device), source_label.to(device), target_csi.to(device), target_label.to(device)
            
            # train generator
            optimizer_G.zero_grad()
            # generate fake target CSI and cheat discriminator
            fake_target_csi = model.G_A2B(source_csi)
            pred_fake_target = model.D_B(fake_target_csi)
            loss_GAN_A2B = criterion_GAN(pred_fake_target, torch.ones_like(pred_fake_target))
            fake_source_csi = model.G_B2A(target_csi)
            pred_fake_source = model.D_A(fake_source_csi)
            loss_GAN_B2A = criterion_GAN(pred_fake_source, torch.ones_like(pred_fake_source))
            # cycle consistency loss
            rec_source_csi = model.G_B2A(fake_target_csi)
            loss_cycle_ABA = criterion_cycle(rec_source_csi, source_csi)
            rec_target_csi = model.G_A2B(fake_source_csi)
            loss_cycle_BAB = criterion_cycle(rec_target_csi, target_csi)
            # total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + args.lambda_cycle * (loss_cycle_ABA + loss_cycle_BAB)
            loss_G.backward()
            optimizer_G.step()
            
            [fake_target_csi, fake_source_csi] = image_pool(
                [fake_target_csi, fake_source_csi])
            
            # train discriminator A
            optimizer_D_A.zero_grad()
            # real loss
            pred_real = model.D_A(source_csi)
            loss_D_real_A = criterion_GAN(pred_real, torch.ones_like(pred_real))
            # fake loss
            pred_fake = model.D_A(fake_source_csi.detach())
            loss_D_fake_A = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            # total loss
            loss_D_A = (loss_D_real_A + loss_D_fake_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # train discriminator B
            optimizer_D_B.zero_grad()
            # real loss
            pred_real = model.D_B(target_csi)
            loss_D_real_B = criterion_GAN(pred_real, torch.ones_like(pred_real))
            # fake loss
            pred_fake = model.D_B(fake_target_csi.detach())
            loss_D_fake_B = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            # total loss
            loss_D_B = (loss_D_real_B + loss_D_fake_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            
            # log losses
            writer.add_scalar('train_gen/loss_G', loss_G.item(), epoch * len(cycle_loader) + batch_idx)
            writer.add_scalar('train_gen/loss_GAN_A2B', loss_GAN_A2B.item(), epoch * len(cycle_loader) + batch_idx)
            writer.add_scalar('train_gen/loss_GAN_B2A', loss_GAN_B2A.item(), epoch * len(cycle_loader) + batch_idx)
            writer.add_scalar('train_gen/loss_cycle_ABA', loss_cycle_ABA.item(), epoch * len(cycle_loader) + batch_idx)
            writer.add_scalar('train_gen/loss_cycle_BAB', loss_cycle_BAB.item(), epoch * len(cycle_loader) + batch_idx)
            writer.add_scalar('train_disc/loss_D_A', loss_D_A.item(), epoch * len(cycle_loader) + batch_idx)
            writer.add_scalar('train_disc/loss_D_B', loss_D_B.item(), epoch * len(cycle_loader) + batch_idx)
            writer.add_scalar('train_disc/loss_D_real_A', loss_D_real_A.item(), epoch * len(cycle_loader) + batch_idx)
            writer.add_scalar('train_disc/loss_D_fake_A', loss_D_fake_A.item(), epoch * len(cycle_loader) + batch_idx)
            writer.add_scalar('train_disc/loss_D_real_B', loss_D_real_B.item(), epoch * len(cycle_loader) + batch_idx)
            writer.add_scalar('train_disc/loss_D_fake_B', loss_D_fake_B.item(), epoch * len(cycle_loader) + batch_idx)
            
            if batch_idx % args.log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(cycle_loader)}, loss_G: {loss_G.item():.4f}, loss_D_A: {loss_D_A.item():.4f}, loss_D_B: {loss_D_B.item():.4f}, loss_GAN_A2B: {loss_GAN_A2B.item():.4f}, loss_GAN_B2A: {loss_GAN_B2A.item():.4f}, loss_cycle_ABA: {loss_cycle_ABA.item():.4f}, loss_cycle_BAB: {loss_cycle_BAB.item():.4f}, loss_D_real_A: {loss_D_real_A.item():.4f}, loss_D_fake_A: {loss_D_fake_A.item():.4f}, loss_D_real_B: {loss_D_real_B.item():.4f}, loss_D_fake_B: {loss_D_fake_B.item():.4f}")

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()
        
        if epoch % args.test_interval == 0:
            # test model
            model.eval()
            # test on trained data and calculate src2tgt L1 error
            test_trained_l1 = 0
            identity_trained_l1 = 0
            for batch_idx, batch_data in enumerate(cycle_loader):
                source_csi, source_label, target_csi, target_label = batch_data
                source_csi, source_label, target_csi, target_label = \
                    source_csi.to(device), source_label.to(device), target_csi.to(device), target_label.to(device)
                
                with torch.no_grad():
                    fake_target_csi = model.G_A2B(source_csi)
                    test_trained_l1 += F.l1_loss(fake_target_csi, target_csi).item()
                    identity_trained_l1 += F.l1_loss(source_csi, target_csi).item()
            test_trained_l1 /= len(cycle_loader)
            identity_trained_l1 /= len(cycle_loader)
            writer.add_scalar('test/trained_L1', test_trained_l1, epoch)
            writer.add_scalar('test/identity_L1', identity_trained_l1, epoch)
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Trained L1: {test_trained_l1:.4f}, Identity L1: {identity_trained_l1:.4f}")
            
            # test on all data and calculate src2tgt L1 error
            test_all_l1 = 0
            identity_all_l1 = 0
            for batch_idx, batch_data in enumerate(cycle_all_loader):
                source_csi, source_label, target_csi, target_label = batch_data
                source_csi, source_label, target_csi, target_label = \
                    source_csi.to(device), source_label.to(device), target_csi.to(device), target_label.to(device)
                
                with torch.no_grad():
                    fake_target_csi = model.G_A2B(source_csi)
                    test_all_l1 += F.l1_loss(fake_target_csi, target_csi).item()
                    identity_all_l1 += F.l1_loss(source_csi, target_csi).item()
            test_all_l1 /= len(cycle_all_loader)
            identity_all_l1 /= len(cycle_all_loader)
            writer.add_scalar('test/all_L1', test_all_l1, epoch)
            writer.add_scalar('test/identity_all_L1', identity_all_l1, epoch)
            logger.info(f"Epoch {epoch+1}/{args.epochs}, All L1: {test_all_l1:.4f}, Identity All L1: {identity_all_l1:.4f}")
            
            # save model
            if test_trained_l1 < min_trained_l1:
                min_trained_l1 = test_trained_l1
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, \
                    f"epoch{epoch+1}_l1trained{test_trained_l1:.4f}_l1all{test_all_l1:.4f}.pth"))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../../0_real_data_preparation/csigan_data', help='path of the dataset')
    parser.add_argument('--epochs',type=int, default=1000, help='# of epoch')
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discri filters in first conv layer')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr_decay_step', type=int, default=50, help='step to do learning rate decay, 0 means no decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight on cycle loss')
    parser.add_argument('--normalize_csi', action='store_true', help='scale csi to [-1, 1]')
    parser.add_argument('--max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
    
    parser.add_argument('--seed', type=int, default=420, help='random seed to use')
    parser.add_argument('--log_dir', default='./cyclegan_logs', help='log are saved here')
    parser.add_argument('--checkpoint_root_dir', default='./cyclegan_checkpoints', help='models are saved here')
    parser.add_argument('--log_interval', type=int, default=10, help='print the debug information every print_freq iterations')
    parser.add_argument('--test_interval', type=int, default=1, help='test the model every test_interval epochs')
    args = parser.parse_args()
    
    main(args)