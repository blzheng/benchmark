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
        self.batchnorm2d57 = BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x301, x296):
        x302=operator.mul(x301, x296)
        x303=self.conv2d97(x302)
        x304=self.batchnorm2d57(x303)
        return x304

m = M().eval()
x301 = torch.randn(torch.Size([1, 768, 1, 1]))
x296 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x301, x296)
end = time.time()
print(end-start)
