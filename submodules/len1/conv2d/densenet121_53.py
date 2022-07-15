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
        self.conv2d53 = Conv2d(480, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x192):
        x193=self.conv2d53(x192)
        return x193

m = M().eval()
x192 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x192)
end = time.time()
print(end-start)
