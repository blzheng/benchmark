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
        self.relu42 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(408, 408, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=17, bias=False)
        self.batchnorm2d47 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x148):
        x149=self.relu42(x148)
        x150=self.conv2d46(x149)
        x151=self.batchnorm2d46(x150)
        x152=self.relu43(x151)
        x153=self.conv2d47(x152)
        x154=self.batchnorm2d47(x153)
        x155=self.relu44(x154)
        x156=self.conv2d48(x155)
        x157=self.batchnorm2d48(x156)
        return x157

m = M().eval()
x148 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x148)
end = time.time()
print(end-start)
