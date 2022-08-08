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
        self.batchnorm2d25 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(408, 408, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=17, bias=False)
        self.batchnorm2d26 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)
        self.conv2d27 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x80):
        x81=self.batchnorm2d25(x80)
        x82=self.relu22(x81)
        x83=self.conv2d26(x82)
        x84=self.batchnorm2d26(x83)
        x85=self.relu23(x84)
        x86=self.conv2d27(x85)
        x87=self.batchnorm2d27(x86)
        return x87

m = M().eval()
x80 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x80)
end = time.time()
print(end-start)
