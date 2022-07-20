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
        self.conv2d92 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x269, x274):
        x275=operator.mul(x269, x274)
        x276=self.conv2d92(x275)
        return x276

m = M().eval()
x269 = torch.randn(torch.Size([1, 864, 14, 14]))
x274 = torch.randn(torch.Size([1, 864, 1, 1]))
start = time.time()
output = m(x269, x274)
end = time.time()
print(end-start)
