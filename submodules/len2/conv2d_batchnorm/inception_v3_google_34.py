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
        self.conv2d34 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x113):
        x126=self.conv2d34(x113)
        x127=self.batchnorm2d34(x126)
        return x127

m = M().eval()
x113 = torch.randn(torch.Size([1, 768, 12, 12]))
start = time.time()
output = m(x113)
end = time.time()
print(end-start)
