from kan import *
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Read .csv file
file_path = 'kc_house_data.csv'
house_data = pd.read_csv(file_path)

# Construct custom dataset class to be used with Torch
class HousePricesDataset(Dataset):
    """House Prices dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.house_data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.house_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        house = self.house_data.iloc[idx, 0:]
        #house['Price'] = np.array([house['Price']])
        #sample = house.to_dict()
        sample = np.array([house['sqft_living'], house['bathrooms'], house['sqft_above'], house['sqft_living15'], house['grade'], house['price']], dtype=np.float32)

        if self.transform:
            sample = self.transform(sample)

        return sample


# split dataset
house_dataset = HousePricesDataset(csv_file=file_path)
data_train, data_test, data_val = torch.utils.data.random_split(house_dataset, [0.7, 0.2, 0.1])

# define dataloader
dataloader_train = DataLoader(data_train, batch_size=5, shuffle=True, num_workers=0)
dataloader_test = DataLoader(data_test, batch_size=len(data_test), shuffle=True, num_workers=0)
dataloader_val = DataLoader(data_val, batch_size=5, shuffle=True, num_workers=0)



# Create Neural Network (KAN)
device = "cuda" if torch.cuda.is_available() else "cpu"

kan_model = KAN(width=[5, 1, 14, 1], grid=5, k=3, seed=2, device=device)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
#loss_fn = nn.L1Loss() # mean absolute error
learning_rate = 0.5
optimizer = optim.Adam(kan_model.parameters(), lr=learning_rate)


# Training session
epochs = 100
batch_start = torch.arange(0, len(dataloader_train))
best_mse = np.inf  # Initialize to infinity
best_weights = None
history_train = []  # Track training loss
history_val = []    # Track validation loss

for epoch in range(epochs):
    kan_model.train()  # Set the model to training mode
    train_loss = 0.0  # Initialize train loss for this epoch
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i_batch, sample_batched in enumerate(dataloader_train):
            sample_batched_X = sample_batched[:, :5].to(device)
            sample_batched_Y = sample_batched[:, 5:].to(device)

            # Forward pass
            pred = kan_model(sample_batched_X)
            loss = loss_fn(pred, sample_batched_Y)

            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training loss
            train_loss += float(loss)  # Accumulate batch loss
            bar.set_postfix(mse=float(loss))  # Show current batch loss in progress bar
    
    # Calculate the average training loss for this epoch
    avg_train_loss = train_loss / len(dataloader_train)
    history_train.append(avg_train_loss)

    # Validation phase (no weight updates, so disable gradients)
    kan_model.eval()  # Set the model to evaluation mode
    val_loss = 0.0  # Initialize validation loss for this epoch
    # with torch.no_grad():  # No gradient calculation needed during validation
    for i_batch, sample_batched in enumerate(dataloader_test):
        sample_batched_X = sample_batched[:, :5].to(device)
        sample_batched_Y = sample_batched[:, 5:].to(device)

        # Forward pass (validation)
        y_pred = kan_model(sample_batched_X)
        mse = loss_fn(y_pred, sample_batched_Y)

        # Accumulate validation loss
        val_loss += float(mse)
    
    # Calculate the average validation loss for this epoch
    avg_val_loss = val_loss / len(dataloader_test)
    history_val.append(avg_val_loss)

    # Print training and validation loss for this epoch
    # print(f"Epoch {epoch + 1}/{epochs} - Training MSE: {avg_train_loss:.4f}, Validation MSE: {avg_val_loss:.4f}")

    # Save the best model based on validation MSE
    if avg_val_loss < best_mse:
        best_mse = avg_val_loss
        # best_weights = copy.deepcopy(kan_model.state_dict())

# # Restore the model with the best weights (based on validation MSE)
# kan_model.load_state_dict(best_weights)

# Print final results
print(f"Best Validation MSE: {best_mse:.4f}")
print(f"Best Validation RMSE: {np.sqrt(best_mse):.4f}")

# Plot training and validation loss over epochs
plt.plot(history_train, label='Training MSE')
plt.plot(history_val, label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training and Validation Loss')
plt.legend()
# plt.show()

# Save the output image
plt.savefig('KANoutput3layer.png')