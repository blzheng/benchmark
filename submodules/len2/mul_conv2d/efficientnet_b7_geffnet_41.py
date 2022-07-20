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
        self.conv2d206 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x610, x615):
        x616=operator.mul(x610, x615)
        x617=self.conv2d206(x616)
        return x617

m = M().eval()
x610 = torch.randn(torch.Size([1, 2304, 7, 7]))
x615 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x610, x615)
end = time.time()
print(end-start)
