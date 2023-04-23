from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
import torch
from tqdm import tqdm
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from util import inference, post_trans

def train(trainloader: DataLoader, testloader: DataLoader, device: torch.device, model: torch.nn.Module, loss_fn: _Loss, optimizer: Optimizer, num_epochs: int, scaler: GradScaler, lr_scheduler: _LRScheduler):
  dice_metric = DiceMetric(include_background=True, reduction="mean")
  hausdorff_distance_metric = HausdorffDistanceMetric(reduction="mean")

  epoch_losses: list[float] = []
  epoch_dice_scores: list[float] = []
  epoch_hausdorff_distance_scores: list[float] = []

  for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch}")

    epoch_loss = 0
    step = 0

    for data in tqdm(trainloader, unit="epoch"):
        step += 1
        labels: torch.Tensor = data['label'].to(device)
        images: torch.Tensor = data['image'].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    
    lr_scheduler.step()
    epoch_loss /= step
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    epoch_losses.append(epoch_loss)

    # Begin testing loop
    print("Beginning Testing...")
    with torch.no_grad():
      for item in tqdm(testloader, unit="evaluation"):
        labels: torch.Tensor = item['label'].to(device)
        images: torch.Tensor = item['image'].to(device)

        # HD metrics can't use zero masks but they are in the dataset
        # so skip them.
        if 0 == labels.count_nonzero():
          continue

        outputs: torch.Tensor = inference(model, images)
        outputs = torch.stack([post_trans(i) for i in decollate_batch(outputs)])
        dice_metric(outputs, labels)
        hausdorff_distance_metric(outputs, labels)

      # Print accuracy
      print(f'Dice Score = {dice_metric.aggregate().item()}')
      print(f'Hausdorff Distance Score = {hausdorff_distance_metric.aggregate().item()}')
      print('--------------------------------')

      epoch_hausdorff_distance_scores.append(hausdorff_distance_metric.aggregate().item())
      epoch_dice_scores.append(dice_metric.aggregate().item())

      hausdorff_distance_metric.reset()
      dice_metric.reset()

  return {
    'epoch_losses': epoch_losses,
    'epoch_dice_scores': epoch_dice_scores,
    'epoch_hausdorff_distance_scores': epoch_hausdorff_distance_scores, 
  }
