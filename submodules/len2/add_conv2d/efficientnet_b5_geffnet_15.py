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
        self.conv2d98 = Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x292, x278):
        x293=operator.add(x292, x278)
        x294=self.conv2d98(x293)
        return x294

m = M().eval()
x292 = torch.randn(torch.Size([1, 128, 14, 14]))
x278 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x292, x278)
end = time.time()
print(end-start)
