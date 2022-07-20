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
        self.conv2d83 = Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x241, x246):
        x247=operator.mul(x241, x246)
        x248=self.conv2d83(x247)
        return x248

m = M().eval()
x241 = torch.randn(torch.Size([1, 672, 14, 14]))
x246 = torch.randn(torch.Size([1, 672, 1, 1]))
start = time.time()
output = m(x241, x246)
end = time.time()
print(end-start)