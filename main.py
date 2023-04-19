from monai.apps import DecathlonDataset
from transforms import train_transform
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from train import train

def visualize(dataset: Dataset):
  val_data_example = dataset[2]
  print(f"image shape: {val_data_example['image'].shape}")
  plt.figure("image", (24, 6))
  for i in range(4):
      plt.subplot(1, 4, i + 1)
      plt.title(f"image channel {i}")
      plt.imshow(val_data_example["image"][i, :, :, 60].detach().cpu(), cmap="gray")
  plt.show()
  # also visualize the 3 channels label corresponding to this image
  print(f"label shape: {val_data_example['label'].shape}")
  plt.figure("label", (18, 6))
  for i in range(3):
      plt.subplot(1, 3, i + 1)
      plt.title(f"label channel {i}")
      plt.imshow(val_data_example["label"][i, :, :, 60].detach().cpu())
  plt.show()

def run_fold(folded_dataset):
  for fold, (train_ids, labels) in enumerate(folded_dataset):
    print(labels)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--num_workers", type=str, default=4, help="The number of workers to use when loading the dataset")
  parser.add_argument("--k_folds", type=int, default=5, help="The number of folds to use for cross validation")

  args = parser.parse_args()

  kfold = KFold(n_splits=args.k_folds, shuffle=True)

  train_dataset = DecathlonDataset(
    root_dir="./data",
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=False,
    num_workers=args.num_workers
  )

  dataset = kfold.split(train_dataset)