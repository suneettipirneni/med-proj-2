from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

from util import inference, post_trans

def visualize(dataset: Dataset, model: torch.nn.Module, device):
  index = 6
  frame = 70
  input = dataset[index]['image'].unsqueeze(0).to(device)
  predicted = post_trans(inference(model, input)[0])
  fig_size = (24, 6)
  num_pred_channels = 3

  plt.figure("Predicted Mask")
  for channel in range(num_pred_channels):
    plt.subplot(1, num_pred_channels, channel + 1)
    plt.title(f"Predicted mask channel {channel}")
    plt.imshow(predicted[channel, :, :, frame].detach().cpu(), cmap="gray")

  plt.savefig("pred.png")

  plt.figure("Ground Truth Mask")
  for channel in range(num_pred_channels):
    plt.subplot(1, num_pred_channels, channel + 1)
    plt.title(f"GT mask channel {channel}")
    plt.imshow(dataset[index]['label'][channel, :, :, frame].detach().cpu(), cmap="gray")
    
  plt.savefig("gt.png") 


