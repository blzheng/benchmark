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
        self.conv2d127 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x379):
        x380=self.conv2d127(x379)
        return x380

m = M().eval()
x379 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x379)
end = time.time()
print(end-start)
