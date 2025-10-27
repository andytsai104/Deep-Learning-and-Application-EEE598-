import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Residual Block: 6 layers (conv*3 + bn*3)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Define shortcut
        self.shortcut = nn.Sequential()
        if (stride != 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    # procedure of residual block
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Define the structure of ResNet
class CustomResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(CustomResNet, self).__init__()
        self.in_channels = 128  # Updated to match the output of conv2

        self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Updated: `layer1` should now take 128 input channels since `conv2` outputs 128
        self.layer1 = self._make_layer(block, 256, num_block[0], stride=1)
        self.layer2 = self._make_layer(block, 512, num_block[1], stride=2)  
        self.layer3 = self._make_layer(block, 1024, num_block[2], stride=2)
        self.layer4 = self._make_layer(block, 2048, num_block[3], stride=2)

        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels  # Update in_channels to match the out_channels of the block
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # First 2 conv layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        out = F.relu(out)

        # Maxpooling
        out = self.maxpool(out)

        # residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global Average Pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)  # Flatten the output for the fully connected layer

        # Fully Connected layer
        out = self.fc(out)
        return out
    
model = CustomResNet(ResidualBlock, [3, 6, 8, 2], num_classes=10)
model_summary = summary(model, input_size=(3, 224, 224))