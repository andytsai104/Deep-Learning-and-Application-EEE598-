import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=11, )
        )
        self.conv1 = nn.Conv2d()
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d()

    


    def forward(self, x):
        
        
        
        
        return x