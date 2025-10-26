from torchvision import datasets, transforms
from ActivationFunction import CustomActivation
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ResNet36 import CustomResNet36, ResidualBlock
import torch
from torch.utils.data import random_split
from torchvision import models
import random

# Define transforms (resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training dataset
ImageNet_path = "/data/datasets/community/deeplearning/imagenet/train"

# Load the ImageNet dataset (assuming dataset is downloaded in 'path_to_imagenet')
dataset = datasets.ImageFolder(ImageNet_path, transform=transform)

# Set random seed for reproducibility
random.seed(42)

# Select a subset of classes
num_class = 500    # choose 500 classes from ImageNet
selected_classes = sorted(random.sample(range(1000), num_class))

# Filter the dataset to include only the selected classes
filtered_dataset = [data_tuple for data_tuple in dataset.samples if data_tuple[1] in selected_classes]

# Randomly sample 600 images per class
image_per_class = 600
final_dataset = []
for class_idx in selected_classes:
    # check the class is selected
    class_images = [sample for sample in filtered_dataset if sample[1] == class_idx]
    
    if len(class_images) < image_per_class:
        sampled_images = class_images
    else:
        sampled_images = random.sample(class_images, image_per_class)
    final_dataset.extend(sampled_images)

# Randomize final_dataset's distribution
random.shuffle(final_dataset)

# Split into training (80%), validation (10%), and test (10%)
train_size = int(0.7 * len(final_dataset))
val_size = int(0.15 * len(final_dataset))
test_size = len(final_dataset) - train_size - val_size

# Random split
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])


# Create DataLoader for batch
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)

model = CustomResNet36(ResidualBlock, [3, 3, 3, 2], num_classes=num_class).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    total = 0
    correct = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Check if any target labels are out of range
        if torch.any(targets >= num_class):
            raise ValueError(f"Invalid target label detected. Max target should be {num_class-1}, but got: {targets}")
        
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
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)} - Loss: {running_loss / (batch_idx + 1):.4f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')

    return running_loss / len(train_loader), 100.*correct / total

# Define the testing function
def test(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    total = 0
    correct = 0
    
    with torch.no_grad():  # No gradient computation during evaluation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(test_loader), 100.*correct / total

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