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
        self.relu73 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(800, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d74 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu74 = ReLU(inplace=True)

    def forward(self, x261):
        x262=self.relu73(x261)
        x263=self.conv2d73(x262)
        x264=self.batchnorm2d74(x263)
        x265=self.relu74(x264)
        return x265

m = M().eval()
x261 = torch.randn(torch.Size([1, 800, 14, 14]))
start = time.time()
output = m(x261)
end = time.time()
print(end-start)
