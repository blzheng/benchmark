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
        self.conv2d204 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x653, x638):
        x654=operator.add(x653, x638)
        x655=self.conv2d204(x654)
        return x655

m = M().eval()
x653 = torch.randn(torch.Size([1, 304, 7, 7]))
x638 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x653, x638)
end = time.time()
print(end-start)
