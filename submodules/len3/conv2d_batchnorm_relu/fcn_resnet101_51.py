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
        self.conv2d79 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d79 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu76 = ReLU(inplace=True)

    def forward(self, x262):
        x263=self.conv2d79(x262)
        x264=self.batchnorm2d79(x263)
        x265=self.relu76(x264)
        return x265

m = M().eval()
x262 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x262)
end = time.time()
print(end-start)
