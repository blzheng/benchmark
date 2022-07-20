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
        self.conv2d48 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x142):
        x143=self.conv2d48(x142)
        x144=self.batchnorm2d28(x143)
        return x144

m = M().eval()
x142 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x142)
end = time.time()
print(end-start)
