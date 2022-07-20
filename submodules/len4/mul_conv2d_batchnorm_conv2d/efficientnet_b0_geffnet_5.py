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
        self.conv2d79 = Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d80 = Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x226, x231):
        x232=operator.mul(x226, x231)
        x233=self.conv2d79(x232)
        x234=self.batchnorm2d47(x233)
        x235=self.conv2d80(x234)
        return x235

m = M().eval()
x226 = torch.randn(torch.Size([1, 1152, 7, 7]))
x231 = torch.randn(torch.Size([1, 1152, 1, 1]))
start = time.time()
output = m(x226, x231)
end = time.time()
print(end-start)
