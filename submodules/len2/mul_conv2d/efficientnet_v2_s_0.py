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
        self.conv2d23 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x76, x71):
        x77=operator.mul(x76, x71)
        x78=self.conv2d23(x77)
        return x78

m = M().eval()
x76 = torch.randn(torch.Size([1, 256, 1, 1]))
x71 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x76, x71)
end = time.time()
print(end-start)
