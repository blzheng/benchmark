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
        self.conv2d39 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x125, x119):
        x126=self.conv2d39(x125)
        x127=self.batchnorm2d39(x126)
        x128=operator.add(x119, x127)
        x129=self.relu36(x128)
        x130=self.conv2d40(x129)
        return x130

m = M().eval()
x125 = torch.randn(torch.Size([1, 408, 14, 14]))
x119 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x125, x119)
end = time.time()
print(end-start)
