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
        self.batchnorm2d8 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x21):
        x22=self.batchnorm2d7(x21)
        x23=self.relu4(x22)
        x24=self.conv2d8(x23)
        x25=self.batchnorm2d8(x24)
        return x25

m = M().eval()
x21 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x21)
end = time.time()
print(end-start)
