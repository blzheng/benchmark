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
        self.conv2d2 = Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
        self.batchnorm2d3 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.conv2d4 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x3):
        x6=self.conv2d2(x3)
        x7=self.batchnorm2d2(x6)
        x8=self.relu1(x7)
        x9=self.conv2d3(x8)
        x10=self.batchnorm2d3(x9)
        x11=self.relu2(x10)
        x12=self.conv2d4(x11)
        x13=self.batchnorm2d4(x12)
        return x13

m = M().eval()
x3 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x3)
end = time.time()
print(end-start)
