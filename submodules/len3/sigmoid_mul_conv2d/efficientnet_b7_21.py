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
        self.conv2d106 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x329, x325):
        x330=self.sigmoid21(x329)
        x331=operator.mul(x330, x325)
        x332=self.conv2d106(x331)
        return x332

m = M().eval()
x329 = torch.randn(torch.Size([1, 960, 1, 1]))
x325 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x329, x325)
end = time.time()
print(end-start)
