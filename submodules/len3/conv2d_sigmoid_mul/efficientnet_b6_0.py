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
        self.conv2d3 = Conv2d(14, 56, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()

    def forward(self, x9, x6):
        x10=self.conv2d3(x9)
        x11=self.sigmoid0(x10)
        x12=operator.mul(x11, x6)
        return x12

m = M().eval()
x9 = torch.randn(torch.Size([1, 14, 1, 1]))
x6 = torch.randn(torch.Size([1, 56, 112, 112]))
start = time.time()
output = m(x9, x6)
end = time.time()
print(end-start)
