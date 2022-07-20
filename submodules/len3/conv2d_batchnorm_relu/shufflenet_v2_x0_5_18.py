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
        self.conv2d28 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)

    def forward(self, x170):
        x171=self.conv2d28(x170)
        x172=self.batchnorm2d28(x171)
        x173=self.relu18(x172)
        return x173

m = M().eval()
x170 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x170)
end = time.time()
print(end-start)