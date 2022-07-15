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
        self.conv2d64 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x210):
        x211=self.conv2d64(x210)
        return x211

m = M().eval()
x210 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x210)
end = time.time()
print(end-start)
