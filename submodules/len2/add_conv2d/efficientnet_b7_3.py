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
        self.conv2d27 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x82, x67):
        x83=operator.add(x82, x67)
        x84=self.conv2d27(x83)
        return x84

m = M().eval()
x82 = torch.randn(torch.Size([1, 48, 56, 56]))
x67 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x82, x67)
end = time.time()
print(end-start)
