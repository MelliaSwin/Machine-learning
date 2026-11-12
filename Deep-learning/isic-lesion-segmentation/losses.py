import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]  (raw outputs from model)
        targets: [B, H, W]    (integer class labels)
        """

        # Convert logits to probabilities
        model_prob = F.sigmoid(logits)
        # From shape: [B,C,H,W] -> [B,H,W]
        model_prob = model_prob.squeeze(1)
        targets = targets.float().squeeze(1)

        # DiceLoss Formula: Dice = 2* Y intersection Y_pred / Y + Y_pred
        intersection = (model_prob * targets).sum(dim=(1, 2))
        union = (model_prob + targets).sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss
