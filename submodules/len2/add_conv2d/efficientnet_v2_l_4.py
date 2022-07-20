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
        self.conv2d9 = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x34, x28):
        x35=operator.add(x34, x28)
        x36=self.conv2d9(x35)
        return x36

m = M().eval()
x34 = torch.randn(torch.Size([1, 64, 56, 56]))
x28 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x34, x28)
end = time.time()
print(end-start)
