from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import StepLR
import tqdm
import numpy as np
from transformers import SwinForImageClassification


# Hyperparameters
input_size = 256
batch_size = 256
lr = 5e-4
weight_decay = 0.05
num_epochs = 100
workers = 8

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_path = '/home/ctsai67/EEE598/FinalProject/train_and_test/train'
test_path = '/home/ctsai67/EEE598/FinalProject/train_and_test/test'

train_dataset = datasets.ImageFolder(train_path, transform=transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)

# Print out the number of classes
num_classes = len(train_dataset.classes)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

# Define model
model_name = "microsoft/swin-tiny-patch4-window7-224"
model = SwinForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes = True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Define training and testing function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = []
    total_img = 0
    correct = 0

    # Training loop
    for imgs, labels in tqdm.tqdm(train_loader, desc='Training', leave=False):
        # Move to device
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward pass
        outputs = model(imgs)
        logits = outputs.logits
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Loss
        running_loss.append(loss.item())
        _, predicted = logits.max(1)
        correct += (predicted == labels).sum().item()
        total_img += labels.size(0)


    avg_loss = sum(running_loss) / len(train_loader)
    accuracy = 100 * correct / total_img

    return avg_loss, accuracy, running_loss


def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total_img = 0


    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(test_loader, desc='Testing', leave=False):
            # Move to device
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass
            outputs = model(imgs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            # Loss
            running_loss = loss.item()
            _, predicted = logits.max(1)
            correct += (predicted == labels).sum().item()
            total_img += labels.size(0)

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total_img

    # return avg_loss, accuracy
    return avg_loss, accuracy


# Record the best accuracy
best_accuracy = 0
best_model_epoch = 0
training_loss = []
test_acc_list = []

# Training loop
for epoch in range(num_epochs):
    print(f'Epoch: {epoch+1}/{num_epochs}')

    # Train the model
    train_loss, train_accuracy, running_loss = train_model(model, train_loader, optimizer, criterion, device)
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.3f}%")
    training_loss.append(train_loss)

    # Testing the model
    test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
    print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_accuracy:.3f}%\n")
    test_acc_list.append(test_accuracy)

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_epoch = epoch+1

        save_path = r"/home/ctsai67/EEE598/FinalProject/ProjectCode/TrainedModels/SwinTransformer.pth"

        # Save the trained model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
        }, save_path)


    # Clean cuda cache
    torch.cuda.empty_cache()

print()
print(f"Best testing accuracy over {num_epochs} epochs is: {best_accuracy:.3f}% in {best_model_epoch} epoch.")


loss_path = r'/home/ctsai67/EEE598/FinalProject/ProjectCode/loss_img/SwinTransformer.png'
x_label = list(range(len(training_loss)))
plt.figure(figsize=(10,5))
plt.title("Training Loss for Swin Transformer-Tiny")
plt.plot(x_label, training_loss, label="Training Loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(loss_path,  bbox_inches='tight')
plt.close()

# Plot Testing Accuracy
plt.figure(figsize=(10, 5))
testimg_path = r'/home/ctsai67/EEE598/FinalProject/ProjectCode/test_acc_img/SwinTransformer.png'
x_label = list(range(1, len(test_acc_list)+1))
plt.title("Testing Accuracy for Swin Transformer-Tiny")
plt.plot(x_label, test_acc_list, label='Testing Accuracy')
plt.xticks(range(min(x_label), max(x_label) + 1))
plt.xlabel('Epochs')
plt.ylabel("Percentage (%)")
plt.legend()
plt.savefig(testimg_path)
plt.close()