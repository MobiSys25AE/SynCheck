import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class Discriminator(nn.Module):
    def __init__(self, input_nc, mid_nc):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, mid_nc, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(mid_nc, mid_nc * 2, kernel_size=4, stride=2, padding=(1,2))
        self.conv3 = nn.Conv2d(mid_nc * 2, mid_nc * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(mid_nc * 4, mid_nc * 8, kernel_size=4, stride=1, padding=2)
        self.conv5 = nn.Conv2d(mid_nc * 8, 1, kernel_size=4, stride=1, padding=1)

        self.instance_norm1 = nn.InstanceNorm2d(mid_nc * 2, affine=True)
        self.instance_norm2 = nn.InstanceNorm2d(mid_nc * 4, affine=True)
        self.instance_norm3 = nn.InstanceNorm2d(mid_nc * 8, affine=True)

    def forward(self, x):
        h0 = F.leaky_relu(self.conv1(x), 0.2)
        h1 = F.leaky_relu(self.instance_norm1(self.conv2(h0)), 0.2)
        h2 = F.leaky_relu(self.instance_norm2(self.conv3(h1)), 0.2)
        h3 = F.leaky_relu(self.instance_norm3(self.conv4(h2)), 0.2)
        h4 = self.conv5(h3)
        return h4


# Residual Block for the Generator
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(dim, affine=True)
        self.instance_norm2 = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        y = F.relu(self.instance_norm1(self.conv1(x)))
        y = self.instance_norm2(self.conv2(y))
        return x + y


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, mid_nc, output_nc):
        super(ResnetGenerator, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, mid_nc, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(mid_nc, mid_nc * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(mid_nc * 2, mid_nc * 4, kernel_size=3, stride=2, padding=1)

        self.instance_norm1 = nn.InstanceNorm2d(mid_nc, affine=True)
        self.instance_norm2 = nn.InstanceNorm2d(mid_nc * 2, affine=True)
        self.instance_norm3 = nn.InstanceNorm2d(mid_nc * 4, affine=True)

        # Define 9 residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(mid_nc * 4) for _ in range(9)]
        )

        self.deconv1 = nn.ConvTranspose2d(mid_nc * 4, mid_nc * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(mid_nc * 2, mid_nc, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(mid_nc, output_nc, kernel_size=7, stride=1, padding=(3,2))

        self.instance_norm4 = nn.InstanceNorm2d(mid_nc * 2, affine=True)
        self.instance_norm5 = nn.InstanceNorm2d(mid_nc, affine=True)
    
    def forward(self, x):
        c1 = F.relu(self.instance_norm1(self.conv1(x)))
        c2 = F.relu(self.instance_norm2(self.conv2(c1)))
        c3 = F.relu(self.instance_norm3(self.conv3(c2)))

        r = self.residual_blocks(c3)

        d1 = F.relu(self.instance_norm4(self.deconv1(r)))
        d2 = F.relu(self.instance_norm5(self.deconv2(d1)))
        pred = torch.tanh(self.conv4(d2))
        
        return pred


class ImagePool:
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp1 = copy.copy(self.images[idx][0])
            self.images[idx][0] = image[0]
            idx = int(np.random.rand() * self.maxsize)
            tmp2 = copy.copy(self.images[idx][1])
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image
