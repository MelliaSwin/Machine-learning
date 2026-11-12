import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from metrics import calculate_metrics, accuracy_score
from typing import Tuple
from torchvision.utils import save_image
import os

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch on the provided training set.
    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader providing training batches.
        criterion (nn.Module): Loss function used for optimization.
        optimizer (optim.Optimizer): Optimizer for model parameter updates.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Average training loss and Accuracy for the epoch.
    """

    # Training mode
    model.train()
    # Add loss over epochs to compute average
    total_train_loss = 0.0
    # All pixel predictions (Probability between 0 and 1)
    # Actual ground truth labels for those predictions (0: background, 1:lesion)
    all_pixel_pred, all_gt_labels = [], []

    # For every batch of images, masks, and image_id's in dataloader
    for imgs, masks, _ in dataloader:
        # Move tensors to GPU
        imgs = imgs.to(device)
        masks = masks.float().to(device)
        # Forward pass -> compute loss -> backprop -> update weights
        optimizer.zero_grad()
        # Shape [B, C, H, W] logits
        outputs = model(imgs)

        # Convert logits to probabilities (Between 0 and 1)
        model_prob = torch.sigmoid(outputs)    # Shape [B, C, H, W]
        loss = criterion(outputs, masks)       # Shape to target [B,H,W] (Dice Loss)

        loss.backward()
        optimizer.step()

        # Running loss per image
        total_train_loss += loss.item() * imgs.size(0)

        # Single threshold: 0.5 for hard mask
        model_final_pred = (model_prob.squeeze(1) >= 0.5).float()

        # Flatten to 1D of pixel values (0 or 1)
        all_pixel_pred.extend(model_final_pred.view(-1).cpu().numpy())
        all_gt_labels.extend(masks.view(-1).cpu().numpy())

    # Compute average loss and accuracy (how many it got right compared to gt)
    avg_loss = total_train_loss / len(dataloader.dataset)
    acc = accuracy_score(all_gt_labels, all_pixel_pred)
    return avg_loss, acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float]:
    """
    Validate the model on the provided validation set and compute detailed metrics.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader providing validation data.
        criterion (nn.Module): Loss function used for evaluation.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[float, float, float, float, float, float]:
            Average validation loss, Accuracy, DICE, IoU, FPR and FNR.
    """

    # Switch to evaluation mode (no weight updates here!)
    model.eval()
    total_val_loss = 0.0
    all_pixel_pred, all_gt_labels = [], []

    # No backprop/weight updates
    with torch.no_grad():
        for imgs, masks, _ in dataloader:
            imgs = imgs.to(device)
            masks = masks.float().to(device)

            # Logits
            outputs = model(imgs)

            # Convert logits to probabilities (Between 0 and 1)
            model_prob = torch.sigmoid(outputs)  # Shape [B, C, H, W]
            # Dice Loss: Outputs (logits) [B, C, H, W];  Targets (Masks: 0 or 1)[B,H,W]
            loss = criterion(outputs, masks)
            total_val_loss += loss.item() * imgs.size(0)

            # Single threshold: 0.5 for hard mask
            model_final_pred = (model_prob.squeeze(1) >= 0.5).float()

            all_pixel_pred.extend(model_final_pred.view(-1).cpu().numpy())
            all_gt_labels.extend(masks.view(-1).cpu().numpy())

    # Compute average loss and accuracy (how many it got right compared to gt)
    avg_loss = total_val_loss / len(dataloader.dataset)
    acc, dice, iou, fpr, fnr = calculate_metrics(all_gt_labels, all_pixel_pred)
    return avg_loss, acc, dice, iou, fpr, fnr

def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_image_dir: str
) -> Tuple[float, float, float, float, float]:
    """
    Test the model on the provided testing set and compute detailed metrics.
    Save the models' prediction with the input and ground truth.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader providing validation data.
        device (torch.device): Device to perform computations on.
        save_image_dir (str): directory to save the image, mask and predicted mask
                              outputs/model version/image_files/FOLD_n/image_id/[image/mask/pred]
    Returns:
        Tuple[float, float, float, float, float]:
            Accuracy, DICE, IoU, FPR and FNR.
    """
    model.eval()
    all_pixel_pred, all_gt_labels = [], []
    os.makedirs(save_image_dir, exist_ok=True)

    with torch.no_grad():
        for imgs, masks, image_ids in dataloader:
            imgs = imgs.to(device)
            masks = masks.float().to(device)

            outputs = model(imgs)
            model_prob = torch.sigmoid(outputs)
            model_final_pred = (model_prob.squeeze(1) >= 0.5).float()

            all_pixel_pred.extend(model_final_pred.view(-1).cpu().numpy())
            all_gt_labels.extend(masks.view(-1).cpu().numpy())

            # Save the image, mask and predicted mask
            for i, image_id in enumerate(image_ids):
                image_root = os.path.join(save_image_dir, f"{image_id}")
                # outputs/model version/image_files/FOLD_n/image_id/[image/mask/pred]
                img_dir = os.path.join(image_root, "image")
                mask_dir = os.path.join(image_root, "mask")
                pred_dir = os.path.join(image_root, "pred")
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(mask_dir, exist_ok=True)
                os.makedirs(pred_dir, exist_ok=True)

                # Tensors for saving
                img_to_save = imgs[i].detach().cpu()
                img_to_save = (img_to_save * 0.5) + 0.5  # Denormalize
                img_to_save = img_to_save.clamp(0, 1)
                mask_to_save = masks[i].detach().cpu()
                pred_to_save = model_final_pred[i].unsqueeze(0).cpu()

                # File names
                img_name = f"{image_id}.png"
                mask_name = f"{image_id}_segmentation.png"
                pred_name = f"{image_id}_predicted.png"

                # Save
                save_image(img_to_save, os.path.join(img_dir, img_name))
                save_image(mask_to_save, os.path.join(mask_dir, mask_name))
                save_image(pred_to_save, os.path.join(pred_dir, pred_name))

    acc, dice, iou, fpr, fnr = calculate_metrics(all_gt_labels, all_pixel_pred)
    return acc, dice, iou, fpr, fnr