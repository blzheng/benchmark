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
        self.conv2d134 = Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d80 = BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x416, x401):
        x417=operator.add(x416, x401)
        x418=self.conv2d134(x417)
        x419=self.batchnorm2d80(x418)
        return x419

m = M().eval()
x416 = torch.randn(torch.Size([1, 272, 7, 7]))
x401 = torch.randn(torch.Size([1, 272, 7, 7]))
start = time.time()
output = m(x416, x401)
end = time.time()
print(end-start)
