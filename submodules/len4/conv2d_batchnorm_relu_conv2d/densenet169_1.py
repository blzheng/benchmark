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
        self.conv2d3 = Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d4 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x14):
        x15=self.conv2d3(x14)
        x16=self.batchnorm2d4(x15)
        x17=self.relu4(x16)
        x18=self.conv2d4(x17)
        return x18

m = M().eval()
x14 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x14)
end = time.time()
print(end-start)
