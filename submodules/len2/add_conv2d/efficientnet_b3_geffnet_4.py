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
        self.conv2d39 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x115, x101):
        x116=operator.add(x115, x101)
        x117=self.conv2d39(x116)
        return x117

m = M().eval()
x115 = torch.randn(torch.Size([1, 48, 28, 28]))
x101 = torch.randn(torch.Size([1, 48, 28, 28]))
start = time.time()
output = m(x115, x101)
end = time.time()
print(end-start)
