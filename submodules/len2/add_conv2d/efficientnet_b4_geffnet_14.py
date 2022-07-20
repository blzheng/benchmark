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
        self.conv2d99 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x293, x279):
        x294=operator.add(x293, x279)
        x295=self.conv2d99(x294)
        return x295

m = M().eval()
x293 = torch.randn(torch.Size([1, 160, 14, 14]))
x279 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x293, x279)
end = time.time()
print(end-start)
