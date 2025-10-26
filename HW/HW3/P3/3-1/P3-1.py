import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import pandas as pd
import random
from collections import Counter
from PIL import Image

# Set random seed for reproducibility
random.seed(42)

# Select a subset of classes
num_class = 500    # choose 500 classes from ImageNet
selected_classes = sorted(random.sample(range(1000), num_class))
print("Selected Classes:", selected_classes, sep=" ", end="\n\n")

# Define transforms (resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training dataset
ImageNet_path = "/data/datasets/community/deeplearning/imagenet/train"

# Load the ImageNet dataset (assuming dataset is downloaded in 'path_to_imagenet')
dataset = datasets.ImageFolder(ImageNet_path, transform=transform)

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
# print("Shuffled Final Dataset:", final_dataset, sep=" ")


# Split into training (80%), validation (10%), and test (10%)
train_size = int(0.7 * len(final_dataset))
val_size = int(0.15 * len(final_dataset))
test_size = len(final_dataset) - train_size - val_size

train_data = final_dataset[:train_size]
val_data = final_dataset[train_size:train_size + val_size]
test_data = final_dataset[train_size + val_size:]

# Summarize Dataset
summary = {
    'Number of Classes': len(selected_classes),
    'Images per Class': image_per_class,
    'Total Images': len(final_dataset),
    'Training Set Size': len(train_data),
    'Validation Set Size': len(val_data),
    'Test Set Size': len(test_data),
}

# Convert to a DataFrame for easier readability
summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Total Number'])

fig, ax = plt.subplots(figsize=(2,2))  # Adjust size as necessary
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, rowLabels=summary_df.index, cellLoc='center', loc='center')

# Adjust table font size and cell scaling
table.auto_set_font_size(True)
table.set_fontsize(9)
table.scale(1.3, 1.3)  # Increase scaling if needed

# Save the summarized table
plt.savefig('subset_detail.png', bbox_inches='tight')


# Make and save a grid of example images
fig, axes = plt.subplots(5, 6, figsize=(10, 6))

num_show = 30
for i, (img_path, label) in enumerate(random.sample(final_dataset, num_show)):
    # Open the image using Pillow
    img = Image.open(img_path)
    
    # Convert the image to a format suitable for matplotlib
    img_np = np.array(img)
    
    # Plot the image on the grid
    axes[i // 6, i % 6].imshow(img_np)
    
    # Set the title for each subplot
    axes[i // 6, i % 6].set_title(f'Class {label}')
    
    # Remove the axes for a cleaner look
    axes[i // 6, i % 6].axis('off')

# Adjust layout to avoid overlapping of titles and plots
plt.tight_layout()

# Save the plot
plt.savefig("subset_images.png", bbox_inches="tight", dpi=300)