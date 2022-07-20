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
        self.conv2d39 = Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x128, x134, x140, x144):
        x145=torch.cat([x128, x134, x140, x144], 1)
        x146=self.conv2d39(x145)
        return x146

m = M().eval()
x128 = torch.randn(torch.Size([1, 112, 14, 14]))
x134 = torch.randn(torch.Size([1, 288, 14, 14]))
x140 = torch.randn(torch.Size([1, 64, 14, 14]))
x144 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x128, x134, x140, x144)
end = time.time()
print(end-start)
