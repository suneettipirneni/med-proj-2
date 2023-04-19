import torch
from torch import nn

# This code is based on the following tutorial online: https://becominghuman.ai/implementing-unet-in-pytorch-8c7e05a121b4
# It has been adapted and some creative liberties have been taken to make it work for this project

class DownConvBlock(nn.Module):

  act: nn.ReLU
  conv1: nn.Conv2d
  conv2: nn.Conv2d
  
  def __init__(self, in_channels: int, out_channels: int) -> None:
    super().__init__()

    self.act = nn.ReLU()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels)
    self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels)

  def forward(self, x: torch.Tensor):
    x = self.conv1(x)
    x = self.act(x)
    x = self.conv2(x)
    x = self.act(x)

    return x

class UNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.down_conv1 = DownConvBlock(1, 64)
    self.down_conv2 = DownConvBlock(64, 128)
    self.down_conv3 = DownConvBlock(128, 256)
    self.down_conv4 = DownConvBlock(256, 512)
    self.down_conv5 = DownConvBlock(512, 1024)

    self.trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.up_conv1 = DownConvBlock(1024, 512)
    self.trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.up_conv2 = DownConvBlock(512, 256)
    self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.up_conv3 = DownConvBlock(256, 128)
    self.trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.up_conv4 = DownConvBlock(128, 64)

    self.out = nn.Conv2d(64, 2, kernel_size=1)

  def crop_tensor(self, a: torch.Tensor, b: torch.Tensor):
    a_size = a.size()[2]
    b_size = b.size()[2]

    delta = b_size - a_size
    delta = delta // 2

    return b[:, :, delta:b_size - delta, delta:b_size - delta]


  def forward(self, x: torch.Tensor):
    x1 = self.down_conv1(x)
    x2 = self.pool(x)
    x3 = self.down_conv2(x)
    x4 = self.pool(x)
    x5 = self.down_conv3(x)
    x6 = self.pool(x)
    x7 = self.down_conv4(x)
    x8 = self.pool(x)
    x9 = self.down_conv5(x)

    x = self.trans1(x9)
    y = self.crop_tensor(x, x7)
    x = self.up_conv1(torch.cat([x,y], 1))

    x = self.trans2(x)
    y = self.crop_tensor(x, x5)
    x = self.up_conv2(torch.cat([x,y], 1))

    x = self.trans3(x)
    y = self.crop_tensor(x, x3)
    x = self.up_conv3(torch.cat([x,y], 1))

    self.trans4(x)
    y = self.crop_tensor(x, x1)
    x = self.up_conv4(torch.cat([x,y], 1))

    x = self.out(x)

    return x