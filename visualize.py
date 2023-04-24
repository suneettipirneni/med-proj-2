from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

from util import inference

def visualize(dataset: Dataset, model: torch.nn.Module):
  frame = 60
  channel = 0
  input = dataset[0]['image']
  predicted = inference(model, input)

  plt.figure("Evaluation")

  plt.subplot(1, 2)
  plt.imshow(predicted[channel, :, :, frame])

  plt.savefig("vis.png")


