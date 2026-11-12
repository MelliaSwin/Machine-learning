# ISIC dataset setup

## Dataset source

The skin lesion images and corresponding ground-truth segmentation masks come from the International Skin Imaging Collaboration (ISIC). The official download page lists the images, ground-truth masks, and license for each challenge release. [ISIC Challenge Datasets](https://challenge.isic-archive.com/data/)

## Download and setup

The `Images/` and `Masks/` folders are intentionally left empty in the repository because of the size of the dataset.

Download and extract the images and ground-truth masks for the same ISIC lesion segmentation release. Create the folders below if they are missing after cloning the repository.

| Folder            | Files to place there                                              |
| ----------------- | ----------------------------------------------------------------- |
| `dataset/Images/` | Input skin lesion images                                          |
| `dataset/Masks/`  | Ground-truth segmentation masks corresponding to the input images |

Keep the filenames expected by the project so the code can match each image to its mask.
