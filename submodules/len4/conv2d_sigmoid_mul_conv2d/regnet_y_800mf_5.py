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
        self.conv2d32 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d33 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x98, x95):
        x99=self.conv2d32(x98)
        x100=self.sigmoid5(x99)
        x101=operator.mul(x100, x95)
        x102=self.conv2d33(x101)
        return x102

m = M().eval()
x98 = torch.randn(torch.Size([1, 80, 1, 1]))
x95 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x98, x95)
end = time.time()
print(end-start)
