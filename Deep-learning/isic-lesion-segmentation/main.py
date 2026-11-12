import os
import numpy as np
import pandas as pd
import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from losses import DiceLoss
from dataset import CSVDataset
from model import UNet
from train import train_one_epoch, validate, test
from config import config_args
from typing import Any


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device:", DEVICE)


def run_training(args: Any) -> None:
   """
   Execute K-fold cross-validation training and evaluation.

   Args:
       args: Object containing configuration parameters such as dataset directories and image settings.
   Returns:
       None. Results and logs are written to disk.
   """
   # Restrict visible GPUs and set the target device
   #os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
   DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Your model is running on {DEVICE}...\n")


   # Set random seeds for reproducibility
   np.random.seed(args.seed)
   torch.manual_seed(args.seed)

   # Load Dataset
   csv_file_path = os.path.join(args.dataset_dir, args.csv_file)  # Store path of dataset's CSV file
   df = pd.read_csv(csv_file_path)                                # Read CSV file into DataFrame (DF)
   os.makedirs(os.path.join(args.output_dir, args.version, "logs"), exist_ok=True)  # Create folder for logs


   # K-Fold CV
   # K-Fold Object:
   kf = KFold(n_splits=args.num_folds,  # Set to 5 (Config.py)
              shuffle=True,             # Shuffle folds (randomized folds)
              random_state=args.seed)   # Reproducible/Repeatable across runs (Config.py)
   fold_results = []                    # Empty list to store metrics (e.g., acc, Dice, IoU) for each fold
   fold = 0                             # Counter that tracks folds


   # Iterate over each fold from K-Fold Object
   # For each fold, train_index and test_index are given row indices [0, 1, 2, 3....]:
   for train_index, test_index in kf.split(df):
       # Row indices (from train_index, test_index) are extracted and image_id is fed to train_df, test_df
       train_df, test_df = (df.iloc[train_index],  # train_df: (e.g., ISIC_0000001)
                            df.iloc[test_index])   # test_df: (e.g., ISIC_0000001)
       # Split training data:
       train_df, val_df = train_test_split(train_df,
                                           test_size=0.2,  # Validation (20%), Train (80%)
                                           random_state=args.seed)


       # Initialize Datasets and Dataloaders (image, mask, image_id)
       # DataFrames converted to PyTorch Tensors (Calls class CSVDataset):
       train_ds = CSVDataset(args, train_df)  # Return: image pixel values, mask pixel values, image_id
       val_ds = CSVDataset(args, val_df)      # Return: image pixel values, mask pixel values, image_id
       test_ds = CSVDataset(args, test_df)    # Return: image pixel values, mask pixel values, image_id


       # is_train  true attribute
       setattr(train_ds, "is_train", True)


       # Use DataLoader to feed batches of tensors (Config.py)
       train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=1)
       val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=1)
       test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=1)


       # Initialize model, loss, optimizer
       model = UNet(num_classes=args.num_classes, backbone='resnet50').to(DEVICE)
       #criterion = DiceLoss()
       # Variant 5: Dice + CE
       # BCELoss: Convert raw logits to probabilities first -> squeeze->[B, H, W]
       criterion = lambda outputs, targets: (
               DiceLoss()(outputs, targets)
               + BCELoss()(torch.sigmoid(outputs), targets))
       optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


       log_records = []


       # Training loop
       for epoch in range(args.epochs):
           train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, DEVICE)
           val_loss, val_acc, val_dice, val_iou, val_fpr, val_fnr = validate(
               model, val_dl, criterion, DEVICE
           )


           log_records.append({
               "fold": fold + 1,
               "epoch": epoch + 1,
               "train_loss": train_loss,
               "train_acc": train_acc,
               "val_loss": val_loss,
               "val_acc": val_acc,
               "val_dice": val_dice,
               "val_iou": val_iou,
               "val_fpr": val_fpr,
               "val_fnr": val_fnr,
           })


           print(
               f"Epoch [{epoch + 1:02d}/{args.epochs}] "
               f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
               f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
               f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | FPR: {val_fpr:.4f} | FNR: {val_fnr:.4f} "
           )


       # Save per-fold logs
       log_df = pd.DataFrame(log_records)
       log_path = os.path.join(args.output_dir, args.version, "logs", f"fold_{fold + 1}_log.csv")
       log_df.to_csv(log_path, index=False)


       # Final test evaluation
       test_acc, test_dice, test_iou, test_fpr, test_fnr\
           = test(model, test_dl, DEVICE,
                  save_image_dir=os.path.join(args.output_dir, args.version, "images", f"FOLD_{fold+1}")
                  )
       fold_results.append({
           "fold": fold + 1,
           "test_acc": test_acc,
           "test_dice": test_dice,
           "test_iou": test_iou,
           "test_fpr": test_fpr,
           "test_fnr": test_fnr,
       })


       # Free memory and delete model to prevent leakage over folds
       del model
       torch.cuda.empty_cache()
       # ADDDED: Increment fold counter
       fold += 1


   # Save aggregated test results
   results_path = os.path.join(args.output_dir, args.version, "testing_results.csv")
   pd.DataFrame(fold_results).to_csv(results_path, index=False)


if __name__ == "__main__":
   args = config_args.parse_args()
   run_training(args)

