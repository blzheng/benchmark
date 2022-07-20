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
        self.conv2d44 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x142, x127):
        x143=operator.add(x142, x127)
        x144=self.conv2d44(x143)
        return x144

m = M().eval()
x142 = torch.randn(torch.Size([1, 128, 14, 14]))
x127 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x142, x127)
end = time.time()
print(end-start)
