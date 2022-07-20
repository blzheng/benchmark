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
        self.conv2d93 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x273, x269):
        x274=x273.sigmoid()
        x275=operator.mul(x269, x274)
        x276=self.conv2d93(x275)
        return x276

m = M().eval()
x273 = torch.randn(torch.Size([1, 1152, 1, 1]))
x269 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x273, x269)
end = time.time()
print(end-start)
