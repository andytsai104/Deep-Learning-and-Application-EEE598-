import numpy as np
from torchvision import datasets, transforms
from ActivationFunction import CustomActivation
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from ResNet36 import CustomResNet36, ResidualBlock
import torch
import os
from PIL import Image

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms (resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training dataset path
ImageNet_train_path = "/data/datasets/community/deeplearning/imagenet/train"
ImageNet_test_path = "/data/datasets/community/deeplearning/imagenet/test"
ImageNet_val_path = "/data/datasets/community/deeplearning/imagenet/val"

# Load the ImageNet dataset
train_data = datasets.ImageFolder(ImageNet_train_path, transform=transform)

# Get the number of classes in the dataset
num_classes = len(train_data.classes)
print(f"Number of classes: {num_classes}")

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, dummy_label=-1):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform
        self.dummy_label = dummy_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel images
        if self.transform:
            image = self.transform(image)
        return image, self.dummy_label  # Return the image and dummy label

# Create DataLoader for batch
train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
test_loader = ImageDataset(dataset=ImageNet_test_path, batch_size=128, shuffle=False)
val_loader = ImageDataset(dataset=ImageNet_val_path, batch_size=128, shuffle=False)

# Define the model, specifying the number of classes in the final layer
model = CustomResNet36(ResidualBlock, [3, 3, 3, 2], num_classes=num_classes).to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)

# Define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    total = 0
    correct = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Check if any target labels are out of range
        if torch.any(targets >= num_classes):
            raise ValueError(f"Invalid target label detected. Max target should be {num_classes-1}, but got: {targets}")
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 5000 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)} - Loss: {running_loss / (batch_idx + 1):.4f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')

    train_loss = running_loss / len(train_loader)
    train_acc = 100.*correct / total
    
    return train_loss, train_acc

# Define the testing function
def test(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    total = 0
    correct = 0
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    test_loss = running_loss/len(test_loader)
    test_acc = 100.*correct/total
    
    return test_loss, test_acc 

# Full training loop
num_epochs = 10

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    # Train the model
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
    
    # Test the model
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    # Validation accuracy
    val_loss, val_acc = test(model, val_loader, criterion, device)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    
    # Save the model
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch+1,
            'train_loss': train_loss,
        }, f"TrainedModels/CustomizedResNet36-{epoch+1}.pth")
    
    # Clear the cache in GPUs
    torch.cuda.empty_cache()