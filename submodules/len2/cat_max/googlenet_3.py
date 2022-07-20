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
        self.maxpool2d8 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)

    def forward(self, x108, x114, x120, x124):
        x125=torch.cat([x108, x114, x120, x124], 1)
        x141=self.maxpool2d8(x125)
        return x141

m = M().eval()
x108 = torch.randn(torch.Size([1, 128, 14, 14]))
x114 = torch.randn(torch.Size([1, 256, 14, 14]))
x120 = torch.randn(torch.Size([1, 64, 14, 14]))
x124 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x108, x114, x120, x124)
end = time.time()
print(end-start)
