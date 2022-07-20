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
        self.conv2d82 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x239, x244):
        x245=operator.mul(x239, x244)
        x246=self.conv2d82(x245)
        return x246

m = M().eval()
x239 = torch.randn(torch.Size([1, 864, 14, 14]))
x244 = torch.randn(torch.Size([1, 864, 1, 1]))
start = time.time()
output = m(x239, x244)
end = time.time()
print(end-start)
