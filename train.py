from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
import torch
from tqdm import tqdm

def train(trainloader: DataLoader, testloader: DataLoader, device: torch.device, model: torch.nn.Module, loss_fn: _Loss, optimizer: Optimizer, num_epochs: int, scaler: GradScaler, lr_scheduler: _LRScheduler):
  for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch}")

    epoch_loss = 0
    step = 0

    for data in tqdm(trainloader):
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

    # Begin testing loop
    total = 0
    correct = 0

    print("Beginning Testing...")
    with torch.no_grad():
      for item in tqdm(testloader):
        labels: torch.Tensor = data['label'].to(device)
        images: torch.Tensor = data['image'].to(device)

        outputs: torch.Tensor = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

      # Print accuracy
      print(f'Accuracy {100.0 * correct / total}')
      print('--------------------------------')

        