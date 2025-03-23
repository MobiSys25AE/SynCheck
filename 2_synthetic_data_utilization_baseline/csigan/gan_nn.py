import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 25 * 5 * 512)
        self.bn1 = nn.BatchNorm1d(25 * 5 * 512)
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 1), \
            padding=(2, 2), output_padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(256)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 3), \
            padding=(2, 1), output_padding=(1,0))
        self.bn3 = nn.BatchNorm2d(128)
        
        self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=(5, 5), stride=(2, 2), \
            padding=(2, 2), output_padding=(1, 1))

    def forward(self, z_seed):
        x = self.fc1(z_seed)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = x.view(-1, 512, 25, 5)  # Reshape to match the expected dimensions for ConvTranspose2d

        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.deconv2(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        output = torch.tanh(self.deconv3(x))
        return output
    
class Discriminator(nn.Module):
    def __init__(self, seed, category=125, mid_channels=32):
        super(Discriminator, self).__init__()
        self.dropout0 = nn.Dropout(0.2)
        
        self.conv1 = nn.Conv2d(3, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=(5, 2), padding=1)
        
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv4 = nn.Conv2d(mid_channels, mid_channels*2, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(mid_channels*2, mid_channels*2, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(mid_channels*2, mid_channels*2, kernel_size=3, stride=(5, 2), padding=1)
        
        self.dropout2 = nn.Dropout(0.5)
        
        self.conv7 = nn.Conv2d(mid_channels*2, mid_channels*2, kernel_size=3, stride=1, padding=0)
        self.nin1 = nn.Conv2d(mid_channels*2, mid_channels*2, kernel_size=1, stride=1, padding=0)
        self.nin2 = nn.Conv2d(mid_channels*2, mid_channels*2, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(6, 6), stride=1)

        self.fc = nn.Linear(mid_channels*2, category)
        self.initialize_weights(seed)
        
    def forward(self, inp):
        x = inp.view(-1, 3, 200, 30)
        x = self.dropout0(x)
        
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        # shape torch.Size([16, 96, 40, 15])
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        # shape torch.Size([16, 192, 8, 8])
        x = self.dropout2(x)
        
        x = F.leaky_relu(self.conv7(x), 0.2)
        x = F.leaky_relu(self.nin1(x), 0.2)
        x = F.leaky_relu(self.nin2(x), 0.2)
        # shape torch.Size([16, 192, 6, 6])
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        intermediate_layer = x
        logits = self.fc(x)    
        return logits, intermediate_layer
    
    def initialize_weights(self, seed):
        torch.manual_seed(seed)
        def init_xavier(m):
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        # for each module in the model, apply the initialization function
        self.apply(init_xavier)

    
    