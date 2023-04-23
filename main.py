from monai.apps import DecathlonDataset
from monai.losses import DiceLoss
from constants import IMAGE_SIZE, NUM_CLASSES
from transforms import train_transform
from monai.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
import argparse
from sklearn.model_selection import KFold
from monai.networks.nets.unetr import UNETR
from train import train
import torch
import json

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
  parser.add_argument("--num_epochs", type=int, default=20, help="The total number of epochs to run for")

  args = parser.parse_args()

  kfold = KFold(n_splits=args.k_folds, shuffle=True)

  train_dataset = DecathlonDataset(
    root_dir="./data",
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=False,
    num_workers=args.num_workers,
    # cache_rate=0.0,
  )

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  folds = kfold.split(train_dataset)

  # We subtract by one because 0 is a class
  model = UNETR(in_channels=4, out_channels=NUM_CLASSES - 1, img_size=IMAGE_SIZE, feature_size=48)
  model.to(device)

  loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
  optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)  

  for fold, (train_ids, label_ids) in enumerate(folds):
    print(f"Fold {fold}")
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(label_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = DataLoader(
                      train_dataset,
                      batch_size=1, sampler=train_subsampler)

    # Even though this references the test data is it sampled randomly
    # so it will not be the same items in the train loader.
    testloader = DataLoader(
      train_dataset,
      batch_size=1,
      sampler=test_subsampler
    )  

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()

    statistics = train(
      trainloader=trainloader, 
      testloader=testloader, 
      device=device,
      model=model,
      loss_fn=loss_function,
      optimizer=optimizer,
      num_epochs=args.num_epochs,
      scaler=scaler,
      lr_scheduler=lr_scheduler
    )

    # Save statistics in json file
    with open(f'fold-{fold}-statistics.json', 'w') as file:
      json.dump(statistics, file)
