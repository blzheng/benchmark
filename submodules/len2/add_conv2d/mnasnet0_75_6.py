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
        self.conv2d36 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x102, x94):
        x103=operator.add(x102, x94)
        x104=self.conv2d36(x103)
        return x104

m = M().eval()
x102 = torch.randn(torch.Size([1, 72, 14, 14]))
x94 = torch.randn(torch.Size([1, 72, 14, 14]))
start = time.time()
output = m(x102, x94)
end = time.time()
print(end-start)
