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
        self.conv2d74 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x228, x213):
        x229=operator.add(x228, x213)
        x230=self.conv2d74(x229)
        return x230

m = M().eval()
x228 = torch.randn(torch.Size([1, 112, 14, 14]))
x213 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x228, x213)
end = time.time()
print(end-start)
