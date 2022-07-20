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
        self.conv2d201 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x599, x595):
        x600=x599.sigmoid()
        x601=operator.mul(x595, x600)
        x602=self.conv2d201(x601)
        return x602

m = M().eval()
x599 = torch.randn(torch.Size([1, 2304, 1, 1]))
x595 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x599, x595)
end = time.time()
print(end-start)
