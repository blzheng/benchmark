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
        self.conv2d52 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x157, x143):
        x158=operator.add(x157, x143)
        x159=self.conv2d52(x158)
        return x159

m = M().eval()
x157 = torch.randn(torch.Size([1, 48, 56, 56]))
x143 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x157, x143)
end = time.time()
print(end-start)
