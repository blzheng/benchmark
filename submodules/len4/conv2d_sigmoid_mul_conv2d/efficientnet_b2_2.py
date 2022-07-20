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
        self.conv2d12 = Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d13 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x36, x33):
        x37=self.conv2d12(x36)
        x38=self.sigmoid2(x37)
        x39=operator.mul(x38, x33)
        x40=self.conv2d13(x39)
        return x40

m = M().eval()
x36 = torch.randn(torch.Size([1, 4, 1, 1]))
x33 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x36, x33)
end = time.time()
print(end-start)
