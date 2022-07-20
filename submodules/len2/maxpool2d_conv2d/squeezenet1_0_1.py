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
        self.maxpool2d1 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2d10 = Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x24):
        x25=self.maxpool2d1(x24)
        x26=self.conv2d10(x25)
        return x26

m = M().eval()
x24 = torch.randn(torch.Size([1, 256, 54, 54]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)
