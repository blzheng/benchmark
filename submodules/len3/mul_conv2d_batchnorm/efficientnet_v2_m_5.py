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
        self.conv2d53 = Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x173, x168):
        x174=operator.mul(x173, x168)
        x175=self.conv2d53(x174)
        x176=self.batchnorm2d41(x175)
        return x176

m = M().eval()
x173 = torch.randn(torch.Size([1, 640, 1, 1]))
x168 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x173, x168)
end = time.time()
print(end-start)
