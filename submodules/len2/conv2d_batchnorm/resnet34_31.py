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
        self.conv2d31 = Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d31 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x99):
        x105=self.conv2d31(x99)
        x106=self.batchnorm2d31(x105)
        return x106

m = M().eval()
x99 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x99)
end = time.time()
print(end-start)
