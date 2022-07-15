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
        self.conv2d28 = Conv2d(244, 244, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x170):
        x171=self.conv2d28(x170)
        return x171

m = M().eval()
x170 = torch.randn(torch.Size([1, 244, 14, 14]))
start = time.time()
output = m(x170)
end = time.time()
print(end-start)
