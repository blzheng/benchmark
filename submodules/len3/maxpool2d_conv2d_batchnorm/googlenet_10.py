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
        self.maxpool2d10 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2d45 = Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x165):
        x166=self.maxpool2d10(x165)
        x167=self.conv2d45(x166)
        x168=self.batchnorm2d45(x167)
        return x168

m = M().eval()
x165 = torch.randn(torch.Size([1, 832, 14, 14]))
start = time.time()
output = m(x165)
end = time.time()
print(end-start)
