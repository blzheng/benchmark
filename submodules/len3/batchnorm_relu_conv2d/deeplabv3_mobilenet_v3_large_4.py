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
        self.batchnorm2d7 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x23):
        x24=self.batchnorm2d7(x23)
        x25=self.relu4(x24)
        x26=self.conv2d8(x25)
        return x26

m = M().eval()
x23 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x23)
end = time.time()
print(end-start)