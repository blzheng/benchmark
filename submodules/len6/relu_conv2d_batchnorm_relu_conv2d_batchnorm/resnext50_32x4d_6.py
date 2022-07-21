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
        self.relu10 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d12 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d13 = Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x38):
        x39=self.relu10(x38)
        x40=self.conv2d12(x39)
        x41=self.batchnorm2d12(x40)
        x42=self.relu10(x41)
        x43=self.conv2d13(x42)
        x44=self.batchnorm2d13(x43)
        return x44

m = M().eval()
x38 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x38)
end = time.time()
print(end-start)
