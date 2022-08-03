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
        self.relu30 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
        self.batchnorm2d46 = BatchNorm2d(576, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(96, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x131):
        x132=self.relu30(x131)
        x133=self.conv2d46(x132)
        x134=self.batchnorm2d46(x133)
        x135=self.relu31(x134)
        x136=self.conv2d47(x135)
        x137=self.batchnorm2d47(x136)
        return x137

m = M().eval()
x131 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x131)
end = time.time()
print(end-start)
