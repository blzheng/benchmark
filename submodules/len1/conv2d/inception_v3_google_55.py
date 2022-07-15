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
        self.conv2d55 = Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)

    def forward(self, x192):
        x193=self.conv2d55(x192)
        return x193

m = M().eval()
x192 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x192)
end = time.time()
print(end-start)
