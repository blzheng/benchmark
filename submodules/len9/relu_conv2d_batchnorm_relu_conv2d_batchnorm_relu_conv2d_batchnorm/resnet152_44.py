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
        self.relu133 = ReLU(inplace=True)
        self.conv2d139 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d139 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu136 = ReLU(inplace=True)
        self.conv2d140 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d140 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d141 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d141 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x459):
        x460=self.relu133(x459)
        x461=self.conv2d139(x460)
        x462=self.batchnorm2d139(x461)
        x463=self.relu136(x462)
        x464=self.conv2d140(x463)
        x465=self.batchnorm2d140(x464)
        x466=self.relu136(x465)
        x467=self.conv2d141(x466)
        x468=self.batchnorm2d141(x467)
        return x468

m = M().eval()
x459 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x459)
end = time.time()
print(end-start)
