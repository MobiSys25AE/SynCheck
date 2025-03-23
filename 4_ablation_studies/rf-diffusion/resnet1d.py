import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_channels, out_channels, stride=1, group=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                    padding=1, bias=False, groups=group)

def conv1x1(in_channels, out_channels, stride=1, group=1):
    """1x1 convolution"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, 
                    bias=False, groups=group)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, base_channels, stride=1, group=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(in_channels, base_channels, stride, group=group)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(base_channels, base_channels, group=group)
        self.bn2 = nn.BatchNorm1d(base_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, base_channels, stride=1, group=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, base_channels, group=group)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.conv2 = conv3x3(base_channels, base_channels, stride, group=group)
        self.bn2 = nn.BatchNorm1d(base_channels)
        self.conv3 = conv1x1(base_channels, base_channels * self.expansion, group=group)
        self.bn3 = nn.BatchNorm1d(base_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels, activity_num=55, seed=42):
        super(ResNet, self).__init__()
        # B*90*1000
        self.mid_channels = 128
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, activity_num)
        self.fc_open = nn.Linear(512 * block.expansion, activity_num*2, bias=False)
        self.initialize_weights(seed)

    def _make_layer(self, block, base_channels, num_blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.mid_channels != base_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.mid_channels, base_channels * block.expansion, stride),
                nn.BatchNorm1d(base_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.mid_channels, base_channels, stride, group, downsample))
        self.mid_channels = base_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.mid_channels, base_channels, group=group))

        return nn.Sequential(*layers)

    def forward(self, x, return_embedding=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        output = self.avg_pool(c4)
        output = output.view(output.size(0), -1)
        if return_embedding:
            return output
        
        output_classify = self.fc(output)
        output_open = self.fc_open(output) 

        return output_classify, output_open

    def initialize_weights(self, seed):
        torch.manual_seed(seed)
        def init_xavier(m):
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        # for each module in the model, apply the initialization function
        self.apply(init_xavier)


       
def resnet18(num_labels, num_mel_bins, seed=42):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], \
        in_channels=num_mel_bins, activity_num=num_labels, seed=seed)


def resnet34(num_labels, num_mel_bins, seed=42):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], \
        in_channels=num_mel_bins, activity_num=num_labels, seed=seed)



def resnet50(num_labels, num_mel_bins, seed=42):
    """ return a ResNet 50 object
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], \
        in_channels=num_mel_bins, activity_num=num_labels, seed=seed)

def resnet101(num_labels, num_mel_bins, seed=42):
    """ return a ResNet 101 object
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], \
        in_channels=num_mel_bins, activity_num=num_labels, seed=seed)


def resnet152(num_labels, num_mel_bins, seed=42):
    """ return a ResNet 152 object
    """
    return ResNet(Bottleneck, [3, 8, 36, 3],\
        in_channels=num_mel_bins, activity_num=num_labels, seed=seed)
