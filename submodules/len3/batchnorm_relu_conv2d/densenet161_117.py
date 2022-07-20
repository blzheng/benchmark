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
        self.batchnorm2d118 = BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu118 = ReLU(inplace=True)
        self.conv2d118 = Conv2d(1200, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x419):
        x420=self.batchnorm2d118(x419)
        x421=self.relu118(x420)
        x422=self.conv2d118(x421)
        return x422

m = M().eval()
x419 = torch.randn(torch.Size([1, 1200, 7, 7]))
start = time.time()
output = m(x419)
end = time.time()
print(end-start)
