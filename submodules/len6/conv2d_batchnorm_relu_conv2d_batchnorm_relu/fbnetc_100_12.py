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
        self.conv2d34 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
        self.batchnorm2d35 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)

    def forward(self, x110):
        x111=self.conv2d34(x110)
        x112=self.batchnorm2d34(x111)
        x113=self.relu23(x112)
        x114=self.conv2d35(x113)
        x115=self.batchnorm2d35(x114)
        x116=self.relu24(x115)
        return x116

m = M().eval()
x110 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x110)
end = time.time()
print(end-start)
