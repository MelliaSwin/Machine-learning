import os
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CSVDataset(Dataset):
    """
    Attributes:
        df (pd.DataFrame): DataFrame containing image file names.
        args: Object containing configuration parameters such as dataset directories and image settings.
        img_transform (transforms.Compose): Composed transformation pipeline applied to each input image.
        mask_transform (transforms.Compose): Composed transformation pipeline applied to each mask.
    """

    def __init__(self, args, df: pd.DataFrame) -> None:
        """
        Initialize the dataset with arguments and data.
        Initialize the image and mask transformation pipeline for pre-processing.


        Args:
            args: Configuration ArgumentParser.
            df (pd.DataFrame): DataFrame containing 'image_id' column.
        """

        self.df = df.reset_index(drop=True)  # Save data table (DF) into object, re-number indices (rows)
        self.args = args  # Store settings from config.py
        # Transformations applied to images
        self.img_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),  # Resize image to 256 x 256 (Config.py)
            transforms.ToTensor(),  # Convert image (0-255 Pixels) -> Tensor (B/W: 0 - 1)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Shift values from [0,1] -> [-1, 1]
        ])

        # Transformations applied to masks
        self.mask_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Retrieve a single image_id from the dataset and use it to open input image and its mask.


        Args:
            i (int): Index of the sample to retrieve.


        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Image tensor of shape (C, H, W) after transformations.
                - Mask tensor of shape (H, W) after transformation, typically 0 or 1 for binary classification.
                - Image ID of type string used to save the prediction output as the same name.
        """

        row = self.df.iloc[i]  # Access i-th row from CSV/DF: row["image_id"] -> An image and its mask
        base_id = str(row["image_id"])  # e.g., ISIC_0000000
        # Define paths (Config.py) to map to image & it's mask
        image_path = os.path.join(self.args.dataset_dir,
                                  self.args.image_dir,
                                  base_id + ".jpg")

        mask_path = os.path.join(self.args.dataset_dir,
                                 self.args.mask_dir,
                                 base_id + "_segmentation.png")
        # Open Files
        img = Image.open(image_path).convert('RGB')  # Load file, convert to 3 color channels
        mask = Image.open(mask_path).convert('L')  # Load file, convert to grayscale (1 channel)

        # Model_5 Augmentation ONLY:
        # is_train returns true (train_ds), val_ds and test_ds will NOT be augmented
        if getattr(self.args, "version", "") == "Model_5" and getattr(self, "is_train", False):
            j = i % 120
            # For every 30th sample, apply augmentation
            if j < 30:
                img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            elif j < 60:
                img = img.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
            elif j < 90:
                img = img.rotate(180, resample=Image.Resampling.BILINEAR)
                mask = mask.rotate(180, resample=Image.Resampling.NEAREST)
            else:
                img = img.rotate(270, expand=True, resample=Image.Resampling.BILINEAR)
                mask = mask.rotate(270, expand=True, resample=Image.Resampling.NEAREST)

        # Transforms
        image = self.img_transform(img)  # Applies our config.py settings (C, H, W)
        mask = self.mask_transform(mask)  # Applies our config.py settings (H, W)
        mask = (mask >= 0.5).float()  # After transform, hard mask (0 or 1)

        return image, mask, row["image_id"]

