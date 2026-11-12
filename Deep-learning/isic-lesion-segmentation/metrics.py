from typing import Sequence, Tuple
import torch
import torch.nn as nn
import numpy as np

def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[float, float, float, float, float]:
    """
    This function computes the following metrics based on ground truth labels and predicted labels:
      - Accuracy
      - DICE
      - IoU
      - FPR
      - FNR

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A tuple containing:
        (accuracy, dice, iou, fpr, fnr)
    """

    eps = 1e-7
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)

    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    fpr = fp / (tn + fp + eps)
    fnr = fn / (tp + fn + eps)

    return acc, dice, iou, fpr, fnr

def confusion_matrix(    
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[int, int, int, int]:
    """
    This function computes the following metrics based on ground truth labels and predicted labels:
      - TN
      - FP
      - FN
      - TP

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A tuple containing:
        (tn, fp, fn, tp)
    """

    yt = torch.as_tensor(y_true)
    yp = torch.as_tensor(y_pred)
    yt = (yt >= 0.5).to(torch.int64).view(-1)
    yp = (yp >= 0.5).to(torch.int64).view(-1)

    # .item() to make  <int>
    # True Positive, if y_predicted == y_true for == 1, then add to tp
    tp = torch.sum((yt == 1)
                   & (yp == 1)).item()

    # True Negative, if y_predicted == y_true for == 0, then add to tn
    tn = torch.sum((yt == 0)
                   & (yp == 0)).item()

    # False Positive, if model predicted 1, but it was actually 0, add to fp
    fp = torch.sum((yt == 0)
                   & (yp == 1)).item()

    # False Negative: if model predicted 0, but it was actually 1, add to fn
    fn = torch.sum((yt == 1)
                   & (yp == 0)).item()

    return tn, fp, fn, tp



def accuracy_score(    
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """
    This function computes the accuracy score based on ground truth labels and predicted labels.

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A float value of the accuracy score
    """
    # Accuracy = Correct Predictions / All Predictions
    yt = torch.as_tensor(y_true)
    yp = torch.as_tensor(y_pred)
    yt = (yt >= 0.5).to(torch.int64).view(-1)
    yp = (yp >= 0.5).to(torch.int64).view(-1)
    acc = (yp == yt).float().mean().item()
    return acc