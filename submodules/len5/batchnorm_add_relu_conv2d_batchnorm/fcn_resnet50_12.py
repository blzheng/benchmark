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
        self.batchnorm2d33 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x109, x102):
        x110=self.batchnorm2d33(x109)
        x111=operator.add(x110, x102)
        x112=self.relu28(x111)
        x113=self.conv2d34(x112)
        x114=self.batchnorm2d34(x113)
        return x114

m = M().eval()
x109 = torch.randn(torch.Size([1, 1024, 28, 28]))
x102 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x109, x102)
end = time.time()
print(end-start)
