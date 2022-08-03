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
        self.conv2d45 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(1152, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
        self.batchnorm2d46 = BatchNorm2d(1152, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x129):
        x130=self.conv2d45(x129)
        x131=self.batchnorm2d45(x130)
        x132=self.relu30(x131)
        x133=self.conv2d46(x132)
        x134=self.batchnorm2d46(x133)
        x135=self.relu31(x134)
        x136=self.conv2d47(x135)
        return x136

m = M().eval()
x129 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x129)
end = time.time()
print(end-start)
