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
        self.conv2d240 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid48 = Sigmoid()
        self.conv2d241 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x756, x753):
        x757=self.conv2d240(x756)
        x758=self.sigmoid48(x757)
        x759=operator.mul(x758, x753)
        x760=self.conv2d241(x759)
        return x760

m = M().eval()
x756 = torch.randn(torch.Size([1, 96, 1, 1]))
x753 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x756, x753)
end = time.time()
print(end-start)
