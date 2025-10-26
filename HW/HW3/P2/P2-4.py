from ModifiedEffi import ModifiedEfficientNetV2S
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import time
import torch
from torchsummary import summary


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root_path = '/home/ctsai67/EEE598/Assignment\ 3/P2'

# Load and Split the data
dataset_train = Flowers102(root=root_path, download=True, split='train', transform=transform)
dataset_test = Flowers102(root=root_path, download=False, split='test', transform=transform)

train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} A100 GPUs")
    
# Define model
model = ModifiedEfficientNetV2S().eval()
model = torch.nn.DataParallel(model)
model = model.to(device)

# Define Loss function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Time tracking
start_time = time.time()

# Train the model
epochs = 2
for epoch in range(epochs):
    model.train()
    running_loss = 0
    total = 0
    correct = 0

    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predict = output.max(1)
        total += label.size(0)
        correct += predict.eq(label).sum().item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")


# Training time
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
print("\n\nLast 7 lines of torchsummary:")

# Print the last 7 lines of the summary
import io
import sys
output = io.StringIO()
sys.stdout = output

# Model summary
summary(model, (3, 224, 224))

# Reset the stdout
sys.stdout = sys.__stdout__

# Get the captured output and split it into lines
output_str = output.getvalue()
output_lines = output_str.split('\n')
# Print the last 7 lines
for line in output_lines[-8:-1]:  # -8 to -1 to avoid last empty line
    print(line)