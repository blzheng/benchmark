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
        self.conv2d25 = Conv2d(176, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x148):
        x149=self.conv2d25(x148)
        return x149

m = M().eval()
x148 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x148)
end = time.time()
print(end-start)
