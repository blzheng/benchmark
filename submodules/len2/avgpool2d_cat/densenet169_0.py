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
        self.avgpool2d0 = AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x50, x58):
        x51=self.avgpool2d0(x50)
        x59=torch.cat([x51, x58], 1)
        return x59

m = M().eval()
x50 = torch.randn(torch.Size([1, 128, 56, 56]))
x58 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x50, x58)
end = time.time()
print(end-start)
