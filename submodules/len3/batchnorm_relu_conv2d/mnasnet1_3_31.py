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
        self.batchnorm2d46 = BatchNorm2d(1488, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(1488, 248, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x133):
        x134=self.batchnorm2d46(x133)
        x135=self.relu31(x134)
        x136=self.conv2d47(x135)
        return x136

m = M().eval()
x133 = torch.randn(torch.Size([1, 1488, 7, 7]))
start = time.time()
output = m(x133)
end = time.time()
print(end-start)
