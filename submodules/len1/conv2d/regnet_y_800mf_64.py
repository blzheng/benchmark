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
        self.conv2d64 = Conv2d(320, 784, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x201):
        x202=self.conv2d64(x201)
        return x202

m = M().eval()
x201 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x201)
end = time.time()
print(end-start)
