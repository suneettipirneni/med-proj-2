from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from util import inference, post_trans
from transforms import val_transform

def visualize(model: torch.nn.Module, device):
  validation_dataset = DecathlonDataset(
    root_dir="./data",
    task="Task01_BrainTumour",
    transform=val_transform,
    section="training",
    download=False,
    num_workers=4,
    cache_rate=0.0,
  )

  index = 6
  frame = 70
  input = validation_dataset[index]['image'].unsqueeze(0).to(device)
  predicted = post_trans(inference(model, input)[0])
  fig_size = (24, 6)
  num_output_channels = 3
  num_input_channels = 4

  plt.figure("input")
  for channel in range(num_input_channels, fig_size):
    plt.subplot(1, num_input_channels, channel + 1)
    plt.title(f"Input channel {channel}")
    plt.imshow(validation_dataset[index]['image'][channel, :, :, frame].detach().cpu(), cmap="gray")
  plt.savefig("inputs.png")

  plt.figure("Predicted Mask", fig_size)
  for channel in range(num_output_channels):
    plt.subplot(1, num_output_channels, channel + 1)
    plt.title(f"Predicted mask channel {channel}")
    plt.imshow(predicted[channel, :, :, frame].detach().cpu())

  plt.savefig("pred.png")

  plt.figure("Ground Truth Mask", fig_size)
  for channel in range(num_output_channels):
    plt.subplot(1, num_output_channels, channel + 1)
    plt.title(f"GT mask channel {channel}")
    print(validation_dataset[index]['label'].shape)
    plt.imshow(validation_dataset[index]['label'][channel, :, :, frame].detach().cpu())
    
  plt.savefig("gt.png") 


