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
        self.conv2d27 = Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d28 = Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d29 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.batchnorm2d29 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x99):
        x103=self.conv2d27(x99)
        x104=self.batchnorm2d27(x103)
        x105=torch.nn.functional.relu(x104,inplace=True)
        x106=self.conv2d28(x105)
        x107=self.batchnorm2d28(x106)
        x108=torch.nn.functional.relu(x107,inplace=True)
        x109=self.conv2d29(x108)
        x110=self.batchnorm2d29(x109)
        x111=torch.nn.functional.relu(x110,inplace=True)
        return x111

m = M().eval()
x99 = torch.randn(torch.Size([1, 288, 25, 25]))
start = time.time()
output = m(x99)
end = time.time()
print(end-start)
