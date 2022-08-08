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
        self.conv2d31 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d32 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d33 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x102):
        x103=self.conv2d31(x102)
        x104=self.batchnorm2d31(x103)
        x105=self.relu28(x104)
        x106=self.conv2d32(x105)
        x107=self.batchnorm2d32(x106)
        x108=self.relu28(x107)
        x109=self.conv2d33(x108)
        return x109

m = M().eval()
x102 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x102)
end = time.time()
print(end-start)
