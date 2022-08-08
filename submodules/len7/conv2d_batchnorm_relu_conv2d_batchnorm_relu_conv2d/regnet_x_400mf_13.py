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
        self.conv2d44 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d45 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d45 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x141):
        x142=self.conv2d44(x141)
        x143=self.batchnorm2d44(x142)
        x144=self.relu40(x143)
        x145=self.conv2d45(x144)
        x146=self.batchnorm2d45(x145)
        x147=self.relu41(x146)
        x148=self.conv2d46(x147)
        return x148

m = M().eval()
x141 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x141)
end = time.time()
print(end-start)
