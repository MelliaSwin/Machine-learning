import argparse

config_args = argparse.ArgumentParser()

#----------------------------------- Dataset Configs -----------------------------------#
config_args.add_argument('--dataset_dir', type = str, default = "Dataset", help = "The root directory of the dataset")
config_args.add_argument('--image_dir', type = str, default = "Images", help = "The root directory of the image files")
config_args.add_argument('--mask_dir', type = str, default = "Masks", help = "The root directory of the mask files")
config_args.add_argument('--csv_file', type = str, default = "dataset.csv", help = "CSV file name")

#----------------------------------- Preprocessing Configs -----------------------------------#
config_args.add_argument('--seed', type = int, default = 42, help = "seed for reproduciability")
config_args.add_argument('--image_size', type = int, default = 256, help = "Resize images to dimention image_size X image_size")
config_args.add_argument('--num_classes', type = int, default = 1, help = "# of classes")

#----------------------------------- Training Configs -----------------------------------#
config_args.add_argument('--epochs', type = int, default = 10, help = "# of epochs")
config_args.add_argument('--num_folds', type = int, default = 5, help = "# of folds")
config_args.add_argument('--batch', type = int, default = 4, help = "batch size")
config_args.add_argument('--patience', type = int, default = 5, help = "# of epochs before early stopping")
config_args.add_argument('--lr', type = float, default = 1e-5, help = "learning rate")
config_args.add_argument('--output_dir', type = str, default = "./outputs", help = "The root directory of the outputs")
config_args.add_argument('--device_name', type = str, default = "2", help = "The available gpu in the cluster, check with nvidia_smi")
config_args.add_argument('--version', type = str, default = "Model_55", help = "The name of the version run (creates a directory based on the name).")
