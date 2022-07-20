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
        self.conv2d56 = Conv2d(16, 384, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d57 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x173, x170):
        x174=self.conv2d56(x173)
        x175=self.sigmoid11(x174)
        x176=operator.mul(x175, x170)
        x177=self.conv2d57(x176)
        x178=self.batchnorm2d33(x177)
        return x178

m = M().eval()
x173 = torch.randn(torch.Size([1, 16, 1, 1]))
x170 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x173, x170)
end = time.time()
print(end-start)
