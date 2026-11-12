# PYTHON VERSION: Python 3.12.3

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import fetch_openml  # Ignore warning


import torch  # Import main tensor library
import torch.nn as nn  # Provides Basic Building Blocks
from torch.utils.data import TensorDataset, DataLoader # DataLoader wraps an iterable around the Dataset
import torch.optim as optim  # Used for optimization


# Load Dataset and assign to X and Y
usps = fetch_openml('usps', version=1)
# X: 9.3k samples, 256 pixels each; Y: (9.3k labels (0-9)
X = usps['data'].to_numpy()
y = usps['target'].astype(int).to_numpy() - 1  # Adjust bounds (0 - 9)
# print(X.shape, y.shape)

# Perform 5-fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)
# print(skf.get_n_splits(X, y))  # Returns Splits

# List
ACCs = []  # List to store all accuracies of each fold

# Loop to get 5 splits (4 folds, 1 validation)
for i, (train_index, val_index) in enumerate(skf.split(X, y)):  # Train_i/Val_i: 1D Array consisting of indices to a row
    # The 4 folds altogether with their corresponding data and label (80% of samples)
    X_train = X[train_index]  # X_train: 2D Array storing selected rows (80% of data, 256 pixel values)
    y_train = y[train_index]  # y_train: 1D Array storing corresponding labels
    # The validation fold with its corresponding data and label (20% of samples)
    X_val = X[val_index]  # X_val: 2D Array storing selected rows (20% of data, 256 pixel values)
    y_val = y[val_index]  # y_train: 1D Array storing corresponding labels

    # Develop Neural Network (Using PyTorch)
    # Convert Data into Tensors (N-Dimensional Arrays) (PyTorch)
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()

    # Wrap into Dataset objects
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # 80% Samples: wrap our features & Labels
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)  # 20% Samples: wrap our features & Labels
    # Develop DataLoaders (Batching, Shuffling, Processing Data) (Large Samples = Larger Batch Size)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Define Model (NN)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),  # 1st Layer:  Make the 256 input features -> 128 Neurons
        nn.ReLU(),  # Activation function (piecewise linear function)
        nn.Linear(128, 64),  # 2nd Layer: 128 Inputs -> 64 neurons (Halving each layer)
        nn.ReLU(),  # Activation function (piecewise linear function)
        nn.Linear(64, 10),  # Output Layer: with 64 inputs, output 10 (0 - 9 digits)
    )
    # Loss Function (Classification -> Cross-Entropy)
    loss_fn = nn.CrossEntropyLoss()  # How wrong is model's prediction is
    # Optimizer: Adjust weights to reduce error
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 0.001 is LR by default

    # List for convergence graphs
    train_losses =[]
    val_losses =[]

    # Training Phase:
    # Inner loop inside folds
    epochs = 50  # Passes through entire dataset
    for epoch in range(1, epochs + 1):
        # Train Model
        model.train()  # Training Mode
        train_loss = 0  # Counter to sum loss
        # Inner Loop inside Epoch
        for x_batch, y_batch in train_loader:  # Loop over training batches (64)
            optimizer.zero_grad()  # Reset gradients with each new batch
            logits = model(x_batch)  # Forward pass -> feed through network -> return score (logits)
            loss = loss_fn(logits, y_batch)  # Compare predicted with true labels
            loss.backward()  # Backpropagation -> compute gradients of loss
            optimizer.step()  # Adjust weights and bias based on gradients
            train_loss += loss.item()  # Add loss to counter
        avg_train_loss = train_loss / len(train_loader)  # After 50 batches, compute avg train_loss (check convergence)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()  # Setting model to evaluation mode, disabling dropout, use stats for batch normalization
        val_loss = 0  # Counter to sum loss
        # Lists to reset per epoch
        correct = 0
        total = 0
        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for x_batch, y_batch in val_loader:  # Loop over val batches
                logits = model(x_batch)  # Forward pass -> feed through network -> return score (logits)
                val_loss += loss_fn(logits, y_batch).item()  # Calculate cross-entropy loss for each individual batch
                # Calculate predictions
                predicted = logits.argmax(dim=1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        # Compute average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        # Compute Accuracy
        val_acc = correct / total

        print(f"Epoch {epoch}/{epochs} "
              f""f"Train Loss: {avg_train_loss:.4f} "
              f""f"Val   Loss: {avg_val_loss:.4f} "
              f""f"Val  Acc: {val_acc:.4f}")
    # Plot Learning Curves
    epochs_range = range(1, epochs + 1)
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {i} Learning Curve (Loss)')
    plt.legend()
    plt.show()


    # Store values in AUC List
    ACCs.append(val_acc)

# Print values
mean_acc = np.mean(ACCs)
std_acc = np.std(ACCs)
print("\n5 Experiments Accuracies:", [f"{acc:.4f}" for acc in ACCs])
print(f"Average: {mean_acc:.4f}")  # Print the mean
print(f"Standard Deviation: {std_acc:.4f}")  # Print the std
print()






