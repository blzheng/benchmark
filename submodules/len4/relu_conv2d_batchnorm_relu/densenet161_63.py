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
        self.relu130 = ReLU(inplace=True)
        self.conv2d130 = Conv2d(1488, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d131 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu131 = ReLU(inplace=True)

    def forward(self, x462):
        x463=self.relu130(x462)
        x464=self.conv2d130(x463)
        x465=self.batchnorm2d131(x464)
        x466=self.relu131(x465)
        return x466

m = M().eval()
x462 = torch.randn(torch.Size([1, 1488, 7, 7]))
start = time.time()
output = m(x462)
end = time.time()
print(end-start)
