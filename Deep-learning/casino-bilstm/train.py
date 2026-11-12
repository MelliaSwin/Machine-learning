"""
train.py

What is done here:
    - Train the model
    - Then evaluate it
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from config import config_args, N_FEATURES
from dataset import CSVDataset
from model import BiLSTMModel

def train():
    args = config_args.parse_args()

    # This will run on my PC, so I will set it to cuda
    if args.device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU is available')
    else:
        device = torch.device('cpu')
        print('On CPU')

    x_tensor, customer_ids, df_model = CSVDataset()

    # Prepare data Loader
    dataset = TensorDataset(x_tensor)                 # Wrap Tensor [2361, 50, 6] -> into Dataset object
    train_loader = DataLoader(dataset,
                              batch_size=args.batch,  # Feed model 64 users at a time
                              shuffle=True)           # Shuffle Users

    # Initialize model and send to GPU (hopefully)
    model = BiLSTMModel(seq_len=args.seq_len,
                        n_features=N_FEATURES,
                        embedding_dim=args.embedding_dim,
                        num_layers=args.num_layers).to(device)

    # Loss Function: Mean Squared Error (Actual - Predicted)^2
    criterion = nn.MSELoss()

    # Using optimizer Adam() to update weights
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    # Begin Training
    print(f"Starting Training for {args.epochs} Epochs...")
    epoch_losses = []

    for epoch in range(args.epochs):
        model.train()       # Training Mode
        running_loss = 0.0  # Accumulating loss

        for batch in train_loader:
            batch_data = batch[0].to(device)              # [Batch, Seq_Len, N_features]

            # Forward Pass:
            reconstruction = model(batch_data)            # Recreate original features for each time step
            loss = criterion(reconstruction, batch_data)  # Calculate the loss

            # Backward Pass
            optimizer.zero_grad()                         # Clear gradients for next batch
            loss.backward()                               # Compute gradient loss for each affecting weight
            optimizer.step()                              # Move the weights in the direction that reduces loss (Update)

            running_loss += loss.item()                   # Accumulate loss across every batch in an epoch

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        # For each Epoch, print the loss
        if (epoch + 1) % 10 == 0 or (epoch+1) == args.epochs:
            print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.6f}')

    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), epoch_losses, label='Training Loss', color='blue')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_line = os.path.join(args.output_dir, 'training_loss.png')

    plt.savefig(plot_line)
    plt.close()

    results_df = evaluation(model, x_tensor, customer_ids, device, args)
    return model, results_df, df_model

def evaluation(model, x_tensor, customer_ids, device, args):
    model.eval()  # Switch to evaluation mode, no more updating weights/learning
    # Create Loader to test our model
    eval_loader = DataLoader(TensorDataset(x_tensor),  # Load our data
                             batch_size=args.batch,    # Place data into batches of 64
                             shuffle=False)            # Mapping Original -> Reconstruction, need CustomerIDs in order
    # Container for results
    anomaly_results = []  # Collects all anomaly results
    batch_i = 0           # Tracks index for Customer list

    with torch.no_grad():
        for batch in eval_loader:
            batch_data = batch[0].to(device)
            reconstruction = model(batch_data)

            # Compute MSE
            raw_error = (reconstruction - batch_data) ** 2              # Calculate Squared Differences

            # Feature-specific errors
            feature_errors = torch.mean(raw_error, dim=1).cpu().numpy()  # Shape: [Batch, 6]

            # Save Results:
            # Map scores back to CustomerIDs
            customers_in_batch = customer_ids[batch_i:batch_i+len(batch_data)]

            for i in range(len(batch_data)):
                # Record
                record = {
                    'CustomerIdentifier': customers_in_batch[i],
                    'Total_Anomaly_Score': np.mean(feature_errors[i]),
                    'Error_Time': feature_errors[i][0],    # Feature 0: Log_Delta (Speed): Time-Based Anomaly (Manic Velocity)
                    'Error_Amount': feature_errors[i][1],  # Feature 1: Log_Amount (Spikes): Amount-Based Anomaly (Loss of Control)
                    'Error_Bank': feature_errors[i][2],    # Feature 2: Is_Bank_Load: Funding Anomaly (Chasing Losses)
                    'Error_Wager': feature_errors[i][3],   # Feature 3: Is_Wager_Load: Gameplay Loop Anomaly (Indecisive Play)
                    'Error_Cashout': feature_errors[i][4], # Feature 4: Is_Wager_Cashout: Withdrawing then immediately redepositing
                    'Error_Decline': feature_errors[i][5]  # Feature 5: Is_Declined (Desperation) Decline-Based Anomaly (Financial Distress)
                }
                anomaly_results.append(record)  # Will contain all anomaly results for EACH user
            batch_i += len(batch_data)          # Next batch of 64

    # Convert to DataFrame
    results_df = pd.DataFrame(anomaly_results)  # Create a table (like spreadsheet)

    # High Risk Threshold (Top 5% most anomalous users)
    threshold = results_df['Total_Anomaly_Score'].quantile(args.risk_quantile)
    results_df['Risk_Group'] = results_df['Total_Anomaly_Score'] > threshold
    # Print Results
    print(f'95% Risk Threshold: {threshold:.4f}')
    print(f'Number of High Risk Users Found: {results_df['Risk_Group'].sum()}')


    return results_df

if __name__ == '__main__':
    train()