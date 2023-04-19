from monai.apps import DecathlonDataset
from transforms import train_transform
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, SubsetRandomSampler
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from train import train
import torch

def run_fold(folds, dataset, device):
  for fold, (train_ids, labels) in enumerate(folds):
    train_subsampler = SubsetRandomSampler(train_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = DataLoader(
                      dataset,
                      batch_size=1, sampler=train_subsampler)

    for data in trainloader:
      label = data['label'].to(device)
      image = data['image'].to(device)

      print(f"image size = {image.shape}")
      print(f"label size = {label.shape}")
        

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

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  folds = kfold.split(train_dataset)
  run_fold(folds, train_dataset, device)
