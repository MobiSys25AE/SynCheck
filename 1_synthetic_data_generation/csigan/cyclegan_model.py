import torch
import torch.nn as nn
from cyclegan_module import Discriminator, ResnetGenerator

class CycleGANModel(nn.Module):
    def __init__(self, args, device):
        super(CycleGANModel, self).__init__()
        self.args = args
        self.device = device
        self.disc_class = Discriminator
        self.gen_class = ResnetGenerator
        self.build_model(args, device)
    
    def build_model(self, args, device):
        self.G_A2B = self.gen_class(args.input_nc, args.ngf, args.output_nc).to(device)
        self.G_B2A = self.gen_class(args.input_nc, args.ngf, args.output_nc).to(device)
        self.D_A = self.disc_class(args.input_nc, args.ndf).to(device)
        self.D_B = self.disc_class(args.input_nc, args.ndf).to(device)