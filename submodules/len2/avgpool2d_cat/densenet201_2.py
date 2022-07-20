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
        self.avgpool2d2 = AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x480, x488):
        x481=self.avgpool2d2(x480)
        x489=torch.cat([x481, x488], 1)
        return x489

m = M().eval()
x480 = torch.randn(torch.Size([1, 896, 14, 14]))
x488 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x480, x488)
end = time.time()
print(end-start)
