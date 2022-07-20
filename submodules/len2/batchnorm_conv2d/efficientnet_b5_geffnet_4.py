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
        self.batchnorm2d81 = BatchNorm2d(304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d138 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x410):
        x411=self.batchnorm2d81(x410)
        x412=self.conv2d138(x411)
        return x412

m = M().eval()
x410 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x410)
end = time.time()
print(end-start)
