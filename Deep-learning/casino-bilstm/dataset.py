"""
dataset.py

What is done here:
    - Sequence creation
    - Preprocess data
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from config import config_args, FEATURE_COLS, N_FEATURES

"""
Sequential Function:
    - Creates a sequence for one customer and their last (--seq_len) transactions
"""
def create_user_sequence(group, feature_cols, seq_len):
    # Group: All transactions (Rows) for ONE user
    data = group[feature_cols].values                           # Grab raw numbers for customer's transaction (row)

    # Handle if the customer contains LESS THAN specified transactions (Rows):
    if len(data) < seq_len:
        padding = np.zeros((seq_len - len(data), N_FEATURES))   # Fill in those missing rows with zeros
        return np.vstack((padding, data))                       # Stack those missing rows ONTOP of real data
    # Else: If there's more than specified transactions:
    else:
        return data[-seq_len:]                                  # Slice the array and take the LAST specified num

"""
Preprocess Function:
    - Load CSV file
    - Prepare/Fix data
    - Normalize data
    - Sequence creation
    
    Notes:
    - Some '#' commented code are there to see if shapes are correct
"""
def CSVDataset():
    # ----------------------------------- Load CSV-----------------------------------#
    args = config_args.parse_args()

    # Load data into dataframe
    df = pd.read_csv(args.csv_file)
    # Shape (138928, 5): ROWS, COL
    # print(df.shape)

    # ----------------------------------- Preprocessing -----------------------------------#
    # Fix Timestamps: Convert text -> date
    df['ReqTimeUTC'] = pd.to_datetime(df['ReqTimeUTC'])

    # Sort the data
    df = df.sort_values(by=['AccountIdentifier', 'ReqTimeUTC']).reset_index(drop=True)
    # print(df.head())

    # Time Variable (Speed of Play)
    df['DeltaTime'] = (df.groupby('AccountIdentifier')['ReqTimeUTC']  # Group rows by EACH user
                       .diff()              # Compute the difference from previous row (timestamp)
                       .dt.total_seconds()  # Convert the gap into seconds
                       .fillna(86400))      # Handle First Transaction, subtract from 24 hours
    # print(df[['AccountIdentifier', 'ReqTimeUTC', 'DeltaTime']].head(10))

    # Logarithms: Scaling values to prevent large gaps from dominating
    # log(1 + x) prevents log(0)
    # Add newest variables to our dataframe using np
    df['Log_Delta'] = np.log1p(df['DeltaTime'])
    df['Log_Amount'] = np.log1p(df['TransactionAmount'])
    # print(df[['DeltaTime', 'Log_Delta']].head(10))
    # print(df[['TransactionAmount', 'Log_Amount']].head(10))

    # One-Hot Encoding: Categorize text as 0 or 1:
    # Level 1 Deposit: Personal Bank ACC -> Digital Wallet
    # print(df['TransactionType'].value_counts())
    df['Is_Bank_Load'] = (df['TransactionType'] == 'LOYALTYCARDCREDITCL').astype(int)
    # print(df[['TransactionType', 'Is_Bank_Load']].head(10))

    # Level 2 Deposit + Withdrawals: Digital Wallet -> <- Wagering ACC
    df['Is_Wager_Load'] = (df['TransactionType'] == 'LOYALTYCARDDEBIT').astype(int)      # Deposits
    df['Is_Wager_Cashout'] = (df['TransactionType'] == 'LOYALTYCARDCREDIT').astype(int)  # Withdrawals
    # print(df[['TransactionType', 'Is_Wager_Load']].head(10))
    # print(df[['TransactionType', 'Is_Wager_Cashout']].head(10))

    # print(df['Status'].value_counts())
    df['Is_Declined'] = (df['Status'] == 'DECLINED').astype(int)  # 0 = Approved, 1 = Declined (True or False)
    # print(df[['Status', 'Is_Declined']].head(10))

    # Normalize (Minimum Value = 0, Maximum Value = 1, range of numbers: 0 - 1)
    scaler = MinMaxScaler()
    df[['Log_Delta', 'Log_Amount']] = scaler.fit_transform(df[['Log_Delta', 'Log_Amount']])
    # print(df[['Log_Delta', 'Log_Amount']])
    # print(df.columns)

    cols = ['AccountIdentifier', 'ReqTimeUTC', 'TransactionAmount', 'DeltaTime'] + FEATURE_COLS
    df_model = df[cols].copy()
    # print(df_model.head(10))

    # ----------------------------------- Create Sequences -----------------------------------#
    seq_len = args.seq_len
    # Now loop through every user
    sequences = df_model.groupby('AccountIdentifier').apply(
        lambda g: create_user_sequence(g, FEATURE_COLS, seq_len))

    # Stack them into a 3D array
    x_np = np.stack(sequences.values)

    # Convert to PyTorch Tensor
    x_tensor = torch.from_numpy(x_np).float()
    # Final Shape: [Num_of_users, 50, 6]
    # print(x_tensor.shape): [2361, 50, 6]

    customer_ids = list(sequences.index)

    return x_tensor, customer_ids, df_model






