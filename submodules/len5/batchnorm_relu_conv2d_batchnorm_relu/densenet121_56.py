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
        self.batchnorm2d116 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu116 = ReLU(inplace=True)
        self.conv2d116 = Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d117 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu117 = ReLU(inplace=True)

    def forward(self, x412):
        x413=self.batchnorm2d116(x412)
        x414=self.relu116(x413)
        x415=self.conv2d116(x414)
        x416=self.batchnorm2d117(x415)
        x417=self.relu117(x416)
        return x417

m = M().eval()
x412 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x412)
end = time.time()
print(end-start)
