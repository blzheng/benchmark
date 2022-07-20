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
        self.batchnorm2d27 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x136, x151):
        x137=self.batchnorm2d27(x136)
        x152=operator.add(x137, x151)
        x153=self.relu36(x152)
        x154=self.conv2d49(x153)
        return x154

m = M().eval()
x136 = torch.randn(torch.Size([1, 336, 14, 14]))
x151 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x136, x151)
end = time.time()
print(end-start)
