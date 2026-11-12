"""
config.py

Argument parser for the Gambling Impulsivity Autoencoder project.
Use this to configure hyperparameters, file paths, and other settings.
"""
import argparse
config_args = argparse.ArgumentParser()

#----------------------------------- Dataset Files-----------------------------------#
config_args.add_argument('--csv_file', type = str, default = "Online_casino_DIB.csv", help = "CSV file name")

#----------------------------------- Preprocessing Configs-----------------------------------#
config_args.add_argument('--seq_len', type = int, default = 50, help = "Number of transactions per user sequence.")
# These features do not change, so no argparse:
FEATURE_COLS = [
    'Log_Delta',
    'Log_Amount',
    'Is_Bank_Load',
    'Is_Wager_Load',
    'Is_Wager_Cashout',
    'Is_Declined'
]
N_FEATURES = len(FEATURE_COLS)

#----------------------------------- Training Configs -----------------------------------#
config_args.add_argument('--epochs', type = int, default = 50, help = "Number of epochs to train the autoencoder.")
config_args.add_argument('--batch', type = int, default = 64, help = "Batch size for training.")
config_args.add_argument('--lr', type = float, default = 3e-4, help = "Learning rate")
config_args.add_argument('--weight_decay', type = float, default = 1e-4, help = "L2 regularization")
config_args.add_argument('--device_name', type = str, default = "cuda", help = "Device: 'cuda' or 'cpu'")

#----------------------------------- Model Configs -----------------------------------#
config_args.add_argument('--embedding_dim', type = int, default = 64, help = "Latent code size (hidden dimension).")
config_args.add_argument('--num_layers', type = int, default = 1, help = "Number of LSTM layers (encoder & decoder).")

#----------------------------------- Evaluation Configs -----------------------------------#
config_args.add_argument('--risk_quantile', type = float, default = 0.95, help = "Threshold quantile for high-risk classification.")
config_args.add_argument('--output_dir', type = str, default = "./outputs", help = "The root directory of the outputs")