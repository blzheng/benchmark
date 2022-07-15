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
        self.conv2d46 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x133):
        x134=self.conv2d46(x133)
        return x134

m = M().eval()
x133 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x133)
end = time.time()
print(end-start)
