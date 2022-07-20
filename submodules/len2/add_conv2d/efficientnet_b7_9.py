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
        self.conv2d62 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x192, x177):
        x193=operator.add(x192, x177)
        x194=self.conv2d62(x193)
        return x194

m = M().eval()
x192 = torch.randn(torch.Size([1, 80, 28, 28]))
x177 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x192, x177)
end = time.time()
print(end-start)
