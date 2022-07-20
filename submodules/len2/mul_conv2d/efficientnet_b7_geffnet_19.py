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
        self.conv2d96 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x282, x287):
        x288=operator.mul(x282, x287)
        x289=self.conv2d96(x288)
        return x289

m = M().eval()
x282 = torch.randn(torch.Size([1, 960, 14, 14]))
x287 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x282, x287)
end = time.time()
print(end-start)
