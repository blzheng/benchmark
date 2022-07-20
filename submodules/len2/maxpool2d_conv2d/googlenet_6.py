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
        self.maxpool2d6 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d26 = Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x85):
        x101=self.maxpool2d6(x85)
        x102=self.conv2d26(x101)
        return x102

m = M().eval()
x85 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x85)
end = time.time()
print(end-start)
