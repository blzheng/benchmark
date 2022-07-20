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
        self.conv2d52 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x152):
        x153=x152.mean((2, 3),keepdim=True)
        x154=self.conv2d52(x153)
        return x154

m = M().eval()
x152 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
