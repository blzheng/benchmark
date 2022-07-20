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
        self.conv2d211 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x662, x657):
        x663=operator.mul(x662, x657)
        x664=self.conv2d211(x663)
        return x664

m = M().eval()
x662 = torch.randn(torch.Size([1, 2304, 1, 1]))
x657 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x662, x657)
end = time.time()
print(end-start)
