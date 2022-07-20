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
        self.conv2d138 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d82 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x404, x409):
        x410=operator.mul(x404, x409)
        x411=self.conv2d138(x410)
        x412=self.batchnorm2d82(x411)
        return x412

m = M().eval()
x404 = torch.randn(torch.Size([1, 1632, 7, 7]))
x409 = torch.randn(torch.Size([1, 1632, 1, 1]))
start = time.time()
output = m(x404, x409)
end = time.time()
print(end-start)
