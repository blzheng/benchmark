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
        self.conv2d13 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x41, x32):
        x42=operator.add(x41, x32)
        x43=self.conv2d13(x42)
        return x43

m = M().eval()
x41 = torch.randn(torch.Size([1, 24, 56, 56]))
x32 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x41, x32)
end = time.time()
print(end-start)
