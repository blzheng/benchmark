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
        self.conv2d17 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x53, x41):
        x54=operator.add(x53, x41)
        x55=self.conv2d17(x54)
        return x55

m = M().eval()
x53 = torch.randn(torch.Size([1, 32, 112, 112]))
x41 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x53, x41)
end = time.time()
print(end-start)
