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
        self.conv2d44 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x137):
        x138=self.conv2d44(x137)
        x139=self.batchnorm2d28(x138)
        return x139

m = M().eval()
x137 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x137)
end = time.time()
print(end-start)
