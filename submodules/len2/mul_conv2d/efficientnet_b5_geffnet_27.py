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
        self.conv2d137 = Conv2d(1056, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x403, x408):
        x409=operator.mul(x403, x408)
        x410=self.conv2d137(x409)
        return x410

m = M().eval()
x403 = torch.randn(torch.Size([1, 1056, 7, 7]))
x408 = torch.randn(torch.Size([1, 1056, 1, 1]))
start = time.time()
output = m(x403, x408)
end = time.time()
print(end-start)
