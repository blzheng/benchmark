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
        self.conv2d59 = Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x171, x167):
        x172=x171.sigmoid()
        x173=operator.mul(x167, x172)
        x174=self.conv2d59(x173)
        x175=self.batchnorm2d35(x174)
        return x175

m = M().eval()
x171 = torch.randn(torch.Size([1, 672, 1, 1]))
x167 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x171, x167)
end = time.time()
print(end-start)
