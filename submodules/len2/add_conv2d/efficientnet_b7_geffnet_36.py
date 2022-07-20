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
        self.conv2d212 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x633, x619):
        x634=operator.add(x633, x619)
        x635=self.conv2d212(x634)
        return x635

m = M().eval()
x633 = torch.randn(torch.Size([1, 384, 7, 7]))
x619 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x633, x619)
end = time.time()
print(end-start)
