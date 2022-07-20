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
        self.conv2d113 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x356, x351):
        x357=operator.mul(x356, x351)
        x358=self.conv2d113(x357)
        return x358

m = M().eval()
x356 = torch.randn(torch.Size([1, 336, 1, 1]))
x351 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x356, x351)
end = time.time()
print(end-start)
