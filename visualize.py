from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

from util import inference, post_trans

def visualize(dataset: Dataset, model: torch.nn.Module, device):
  frame = 60
  channel = 0
  input = dataset[0]['image'].unsqueeze(0).to(device)
  predicted = post_trans(inference(model, input)[0])

  plt.imshow(predicted[channel, :, :, frame].detach().cpu(), cmap="gray")
  plt.savefig("vis.png")


