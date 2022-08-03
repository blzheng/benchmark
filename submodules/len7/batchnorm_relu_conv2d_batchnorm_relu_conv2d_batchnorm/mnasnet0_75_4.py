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
        self.batchnorm2d12 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(72, 72, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=72, bias=False)
        self.batchnorm2d13 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(72, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(32, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x35):
        x36=self.batchnorm2d12(x35)
        x37=self.relu8(x36)
        x38=self.conv2d13(x37)
        x39=self.batchnorm2d13(x38)
        x40=self.relu9(x39)
        x41=self.conv2d14(x40)
        x42=self.batchnorm2d14(x41)
        return x42

m = M().eval()
x35 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x35)
end = time.time()
print(end-start)
