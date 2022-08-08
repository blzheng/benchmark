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
        self.conv2d97 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d98 = Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x288, x284, x278):
        x289=x288.sigmoid()
        x290=operator.mul(x284, x289)
        x291=self.conv2d97(x290)
        x292=self.batchnorm2d57(x291)
        x293=operator.add(x292, x278)
        x294=self.conv2d98(x293)
        x295=self.batchnorm2d58(x294)
        return x295

m = M().eval()
x288 = torch.randn(torch.Size([1, 768, 1, 1]))
x284 = torch.randn(torch.Size([1, 768, 14, 14]))
x278 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x288, x284, x278)
end = time.time()
print(end-start)
