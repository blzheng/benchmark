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
        self.conv2d34 = Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)

    def forward(self, x109):
        x110=self.conv2d34(x109)
        x111=self.batchnorm2d34(x110)
        x112=self.relu31(x111)
        return x112

m = M().eval()
x109 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)
