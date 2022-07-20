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
        self.conv2d47 = Conv2d(32, 512, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d48 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x152, x149):
        x153=self.conv2d47(x152)
        x154=self.sigmoid5(x153)
        x155=operator.mul(x154, x149)
        x156=self.conv2d48(x155)
        x157=self.batchnorm2d36(x156)
        return x157

m = M().eval()
x152 = torch.randn(torch.Size([1, 32, 1, 1]))
x149 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x152, x149)
end = time.time()
print(end-start)
