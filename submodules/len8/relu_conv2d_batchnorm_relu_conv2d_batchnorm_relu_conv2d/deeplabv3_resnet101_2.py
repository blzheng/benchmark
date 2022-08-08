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
        self.relu7 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d13 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x37):
        x38=self.relu7(x37)
        x39=self.conv2d11(x38)
        x40=self.batchnorm2d11(x39)
        x41=self.relu10(x40)
        x42=self.conv2d12(x41)
        x43=self.batchnorm2d12(x42)
        x44=self.relu10(x43)
        x45=self.conv2d13(x44)
        return x45

m = M().eval()
x37 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
