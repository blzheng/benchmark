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
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2d3 = Conv2d(64, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x22):
        x23=self.maxpool2d0(x22)
        x24=self.conv2d3(x23)
        return x24

m = M().eval()
x22 = torch.randn(torch.Size([1, 64, 109, 109]))
start = time.time()
output = m(x22)
end = time.time()
print(end-start)
