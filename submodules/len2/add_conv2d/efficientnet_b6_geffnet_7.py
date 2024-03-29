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
        self.conv2d53 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x158, x144):
        x159=operator.add(x158, x144)
        x160=self.conv2d53(x159)
        return x160

m = M().eval()
x158 = torch.randn(torch.Size([1, 72, 28, 28]))
x144 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x158, x144)
end = time.time()
print(end-start)
