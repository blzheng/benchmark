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
        self.conv2d23 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()

    def forward(self, x67, x64):
        x68=self.conv2d23(x67)
        x69=self.sigmoid4(x68)
        x70=operator.mul(x69, x64)
        return x70

m = M().eval()
x67 = torch.randn(torch.Size([1, 10, 1, 1]))
x64 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x67, x64)
end = time.time()
print(end-start)
