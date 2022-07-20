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
        self.sigmoid21 = Sigmoid()
        self.conv2d108 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x333, x329):
        x334=self.sigmoid21(x333)
        x335=operator.mul(x334, x329)
        x336=self.conv2d108(x335)
        return x336

m = M().eval()
x333 = torch.randn(torch.Size([1, 960, 1, 1]))
x329 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x333, x329)
end = time.time()
print(end-start)
