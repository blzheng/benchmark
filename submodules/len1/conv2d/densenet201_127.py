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
        self.conv2d127 = Conv2d(1664, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x451):
        x452=self.conv2d127(x451)
        return x452

m = M().eval()
x451 = torch.randn(torch.Size([1, 1664, 14, 14]))
start = time.time()
output = m(x451)
end = time.time()
print(end-start)
