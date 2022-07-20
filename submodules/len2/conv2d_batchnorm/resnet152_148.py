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
        self.conv2d148 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d148 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x480):
        x489=self.conv2d148(x480)
        x490=self.batchnorm2d148(x489)
        return x490

m = M().eval()
x480 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x480)
end = time.time()
print(end-start)
