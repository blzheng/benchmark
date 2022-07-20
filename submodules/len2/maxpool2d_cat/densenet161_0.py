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
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x3, x11):
        x4=self.maxpool2d0(x3)
        x12=torch.cat([x4, x11], 1)
        return x12

m = M().eval()
x3 = torch.randn(torch.Size([1, 96, 112, 112]))
x11 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x3, x11)
end = time.time()
print(end-start)