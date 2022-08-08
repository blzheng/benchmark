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
        self.conv2d130 = Conv2d(336, 888, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d80 = BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu101 = ReLU(inplace=True)
        self.conv2d131 = Conv2d(888, 888, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=37, bias=False)
        self.batchnorm2d81 = BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu102 = ReLU(inplace=True)

    def forward(self, x409):
        x412=self.conv2d130(x409)
        x413=self.batchnorm2d80(x412)
        x414=self.relu101(x413)
        x415=self.conv2d131(x414)
        x416=self.batchnorm2d81(x415)
        x417=self.relu102(x416)
        return x417

m = M().eval()
x409 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x409)
end = time.time()
print(end-start)
