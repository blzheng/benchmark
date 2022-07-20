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
        self.relu87 = ReLU(inplace=True)
        self.conv2d87 = Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x310):
        x311=self.relu87(x310)
        x312=self.conv2d87(x311)
        return x312

m = M().eval()
x310 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x310)
end = time.time()
print(end-start)
