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
        self.batchnorm2d33 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d35 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x109):
        x110=self.batchnorm2d33(x109)
        x111=self.relu31(x110)
        x112=self.conv2d34(x111)
        x113=self.batchnorm2d34(x112)
        x114=self.relu31(x113)
        x115=self.conv2d35(x114)
        return x115

m = M().eval()
x109 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)
