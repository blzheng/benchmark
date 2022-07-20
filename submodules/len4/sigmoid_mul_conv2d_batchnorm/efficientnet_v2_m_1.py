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
        self.sigmoid1 = Sigmoid()
        self.conv2d33 = Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x108, x104):
        x109=self.sigmoid1(x108)
        x110=operator.mul(x109, x104)
        x111=self.conv2d33(x110)
        x112=self.batchnorm2d29(x111)
        return x112

m = M().eval()
x108 = torch.randn(torch.Size([1, 640, 1, 1]))
x104 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x108, x104)
end = time.time()
print(end-start)
