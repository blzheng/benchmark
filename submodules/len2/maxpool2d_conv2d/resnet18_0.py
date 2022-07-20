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
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2d1 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x3):
        x4=self.maxpool2d0(x3)
        x5=self.conv2d1(x4)
        return x5

m = M().eval()
x3 = torch.randn(torch.Size([1, 64, 112, 112]))
start = time.time()
output = m(x3)
end = time.time()
print(end-start)
