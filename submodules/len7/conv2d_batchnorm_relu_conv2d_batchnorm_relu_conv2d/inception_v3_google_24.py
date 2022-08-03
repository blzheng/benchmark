import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d89 = Conv2d(2048, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d89 = BatchNorm2d(448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d90 = Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d90 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d91 = Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)

    def forward(self, x292):
        x306=self.conv2d89(x292)
        x307=self.batchnorm2d89(x306)
        x308=torch.nn.functional.relu(x307,inplace=True)
        x309=self.conv2d90(x308)
        x310=self.batchnorm2d90(x309)
        x311=torch.nn.functional.relu(x310,inplace=True)
        x312=self.conv2d91(x311)
        return x312

m = M().eval()
x292 = torch.randn(torch.Size([1, 2048, 5, 5]))
start = time.time()
output = m(x292)
end = time.time()
print(end-start)
