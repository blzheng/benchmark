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
        self.relu36 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(1344, 1344, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
        self.batchnorm2d41 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)
        self.conv2d42 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x128):
        x129=self.relu36(x128)
        x130=self.conv2d40(x129)
        x131=self.batchnorm2d40(x130)
        x132=self.relu37(x131)
        x133=self.conv2d41(x132)
        x134=self.batchnorm2d41(x133)
        x135=self.relu38(x134)
        x136=self.conv2d42(x135)
        return x136

m = M().eval()
x128 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x128)
end = time.time()
print(end-start)
