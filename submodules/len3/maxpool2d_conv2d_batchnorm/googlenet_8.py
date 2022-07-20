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
        self.maxpool2d8 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d38 = Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x125):
        x141=self.maxpool2d8(x125)
        x142=self.conv2d38(x141)
        x143=self.batchnorm2d38(x142)
        return x143

m = M().eval()
x125 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x125)
end = time.time()
print(end-start)
