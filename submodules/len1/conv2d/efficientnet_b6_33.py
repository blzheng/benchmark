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
        self.conv2d33 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x102):
        x103=self.conv2d33(x102)
        return x103

m = M().eval()
x102 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x102)
end = time.time()
print(end-start)