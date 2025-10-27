import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Autoencoder_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Autoencoder_Classifier, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Input: 3xHxW (e.g., RGB images)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample by 2
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Further downsample
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),  # Flatten the output
            nn.Linear(64 * (input_size // 4) * (input_size // 4), 128),  # Compress
            nn.ReLU(),
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Output: num_classes
            nn.Softmax(dim=1)  # Convert logits to probabilities
        )
        
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Bottleneck
        latent = self.bottleneck(encoded)
        
        # Classifier
        output = self.classifier(latent)
        
        return output

# Define parameters
input_size = 64  # Assuming 32x32 images (e.g., CIFAR-10)
num_classes = 10  # Number of output classes

# Instantiate the model
model = CNN_Autoencoder_Classifier(num_classes)

# Print model summary
print(model)
